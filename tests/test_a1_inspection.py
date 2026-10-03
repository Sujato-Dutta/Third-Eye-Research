"""Inspection must stop corrupted pilot labels and require evidence-bound review."""

import importlib.util
import json
from pathlib import Path

import pytest

from third_eye.config import Config
from third_eye.data.schema import Correction, Example
from third_eye.io import digest, file_digest, write_json
from third_eye.updates.candidates import batch_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "a1_inspection", ROOT / "cluster/a1_inspection.py"
)
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)


def pilots(tmp_path, terminal=False):
    tasks = []
    for stream in sorted(inspection.STREAMS):
        model, task = stream.split(":")
        cfg = Config.load(ROOT / f"experiments/configs/{model}_a1_pilot.json")
        output = tmp_path / f"{model}_{task}"
        config_path = output / "config.json"
        write_json(config_path, cfg.to_dict())
        write_json(
            output / "run.json",
            {
                "config": cfg.to_dict(),
                "config_hash": cfg.fingerprint,
                "manifest_hash": "data",
            },
        )
        write_json(
            output / "completed.json",
            {"status": "complete", "completed_states": 1, "requested_generations": 1},
        )
        examples = [
            Example(str(i), f"problem {i}", "solution", "train", task) for i in range(3)
        ]
        items = [Correction(ex, "verified solution", 1).to_dict() for ex in examples]
        stats = {
            "harvest_complete": True,
            "correction_attempt_limit": 16,
            "verified_corrections": 3,
            "training_prompts": 3,
        }
        state = output / "states/state"
        write_json(state / "correction_pool.json", {"items": items, "stats": stats})
        rows = []
        for index, pair in enumerate(((0, 1), (0, 2), (1, 2))):
            directory = state / f"k{index}"
            batch = [items[i] for i in pair]
            continuation_items = [] if terminal else items
            write_json(directory / "batch.json", batch)
            write_json(
                directory / "continuation_pool.json",
                {
                    "items": continuation_items,
                    "stats": dict(stats, verified_corrections=len(continuation_items)),
                },
            )
            continuation_batch = [] if terminal else batch
            write_json(directory / "continuation_batch.json", continuation_batch)

            def hashed(values):
                return batch_hash(
                    [
                        Correction(
                            Example(**v["example"]), v["completion"], v["attempt"]
                        )
                        for v in values
                    ]
                )

            parent = {"target": 0.5, "ood": 0.4, "retention": 0.8}
            t1 = {"target": 0.6, "ood": 0.4, "retention": 0.7}
            t2 = t1 if terminal else {"target": 0.7, "ood": 0.5, "retention": 0.7}
            labels = {}
            for horizon, point in ((1, t1), (2, t2)):
                values = {h: point[h] - parent[h] for h in inspection.HEADS}
                labels[f"h{horizon}"] = values
                labels[f"utility_h{horizon}"] = sum(
                    w * values[h]
                    for w, h in zip(cfg.protocol.utility_weights, inspection.HEADS)
                )
            rows.append(
                {
                    "state_id": "state",
                    "candidate_id": f"k{index}",
                    "trajectory_id": stream,
                    "model": cfg.model.name,
                    "seed": 1042,
                    "generation": 0,
                    "config_hash": cfg.fingerprint,
                    "manifest_hash": "data",
                    "parent_adapter_hash": "parent",
                    "protocol_amendment": "A1",
                    "candidate_size": 2,
                    "batch_hash": hashed(batch),
                    "continuation_available": int(not terminal),
                    "evaluation": {"parent": parent, "t1": t1, "t2": t2},
                    "labels": labels,
                    "runtime": {
                        "candidate": {"optimizer_steps": 50},
                        "continuation": {"optimizer_steps": 0 if terminal else 50},
                    },
                    "continuation": {
                        "available": int(not terminal),
                        "terminal_reason": "correction_scarcity" if terminal else None,
                        "batch_hash": None if terminal else hashed(batch),
                        "pool_statistics": dict(
                            stats, verified_corrections=len(continuation_items)
                        ),
                    },
                }
            )
        path = output / "meta_labels.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        tasks.append(
            {
                "stream": stream,
                "seed": 1042,
                "output": str(output),
                "commands": [["experiments/run.py", "--config", str(config_path)]],
            }
        )
    return tasks


def test_valid_pilots_require_evidence_bound_review(tmp_path):
    report = inspection.inspect_pilots(pilots(tmp_path))
    assert report["status"] == "valid" and report["candidate_branches"] == 12
    assert report["terminal_fraction"] == 0
    path = tmp_path / "inspection.json"
    write_json(path, report)
    assert not inspection.review_allows_calibration(report, path, None)
    assert not inspection.review_allows_calibration(report, path, "stale")
    assert inspection.review_allows_calibration(report, path, file_digest(path))


def test_terminal_dominance_is_reported_and_held_without_new_cutoff(tmp_path):
    report = inspection.inspect_pilots(pilots(tmp_path, terminal=True))
    assert report["status"] == "valid" and report["terminal_fraction"] == 1
    assert len(report["all_terminal_streams"]) == 4
    path = tmp_path / "inspection.json"
    write_json(path, report)
    assert not inspection.review_allows_calibration(report, path, None)


@pytest.mark.parametrize(
    "defect",
    [
        "incomplete",
        "missing_metric",
        "wrong_h2",
        "duplicate_batch",
        "terminal_training",
    ],
)
def test_corrupted_pilot_cannot_release_calibration(tmp_path, defect):
    tasks = pilots(tmp_path, terminal=True)
    path = Path(tasks[0]["output"]) / "meta_labels.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "incomplete":
        rows.pop()
    elif defect == "missing_metric":
        rows[0]["evaluation"]["t2"].pop("ood")
    elif defect == "wrong_h2":
        rows[0]["labels"]["h2"]["target"] += 0.1
    elif defect == "terminal_training":
        rows[0]["runtime"]["continuation"]["optimizer_steps"] = 50
    else:
        state = Path(tasks[0]["output"]) / "states/state"
        batch = json.loads((state / "k0/batch.json").read_text())
        write_json(state / "k1/batch.json", batch)
        rows[1]["batch_hash"] = rows[0]["batch_hash"]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    report = inspection.inspect_pilots(tasks)
    assert report["status"] == "inspection_failed" and report["errors"]
    report_path = tmp_path / "inspection.json"
    write_json(report_path, report)
    assert not inspection.review_allows_calibration(
        report, report_path, file_digest(report_path)
    )


def test_review_detects_modified_evidence(tmp_path):
    tasks = pilots(tmp_path)
    before = inspection.inspect_pilots(tasks)
    path = Path(tasks[0]["output"]) / "completed.json"
    data = json.loads(path.read_text())
    write_json(path, dict(data, diagnostic="changed"))
    after = inspection.inspect_pilots(tasks)
    assert after["status"] == "valid" and digest(after) != digest(before)
