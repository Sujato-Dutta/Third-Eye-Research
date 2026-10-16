"""Real tiny adapters test collection contracts; fixtures are not research evidence."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from test_a1 import amended
from third_eye.data.schema import Correction, Example, ROLES
from third_eye.evaluation.runner import evaluate
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.labeling import LabelGenerator, run_trajectory
from third_eye.forecasting.continuation import candidate_inputs, repeated_targets


@pytest.mark.parametrize("terminal", [False, True])
def test_five_draws_capture_items_and_preserve_real_adapters(tmp_path, terminal):
    from verify_backend import make_tiny_backend

    backend = make_tiny_backend(os.environ.get("THIRD_EYE_REPEAT_VERIFY_DEVICE", "cpu"))
    backend.config = amended(backend.config, "A2")
    cfg = backend.config
    splits = {
        role: [
            Example(f"{role}-{i}", f"one plus one {role} {i}", "#### 2", role)
            for i in range(6 if role == "train" else 2)
        ]
        for role in ROLES
    }
    pool = [Correction(ex, "#### 2", 1) for ex in splits["train"]]
    parent = backend.state_hash()
    starts = []
    original_train = backend.train

    def train(*args, **kwargs):
        starts.append(backend.state_hash())
        return original_train(*args, **kwargs)

    def harvest(b, *args):
        available = [] if terminal and b.state_hash() != parent else pool
        return available, {
            "verified_corrections": len(available),
            "harvest_complete": True,
        }

    from dataclasses import asdict
    from third_eye.data.splits import create_manifest
    from third_eye.io import digest, write_json
    from third_eye.provenance import source_inventory

    paths = {}
    for role, examples in splits.items():
        paths[role] = tmp_path / f"{role}.jsonl"
        paths[role].write_text("".join(json.dumps(asdict(e)) + "\n" for e in examples))
    manifest = create_manifest(paths, tmp_path / "selection.json")
    labeler = LabelGenerator(
        backend,
        cfg,
        splits,
        VerifierRegistry(),
        tmp_path,
        digest(manifest),
        continuation_seed_offsets=(100, 200, 300, 400),
        capture_item_outcomes=True,
    )
    with (
        patch.object(backend, "train", train),
        patch("third_eye.experiments.labeling.collect_corrections", harvest),
        patch("third_eye.experiments.continuations.collect_corrections", harvest),
    ):
        accepted = run_trajectory(labeler, generations=1)
    rows = [
        json.loads(line)
        for line in (tmp_path / "meta_labels.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    repeated_targets(rows)
    candidate_inputs(rows)  # Captured outcomes are never input features.
    for index, row in enumerate(rows):
        draws = row["continuation_replicates"]
        assert len(draws) == len({d["seed"] for d in draws}) == 5
        assert row["runtime"]["candidate"]["optimizer_steps"] == 3
        baseline = json.loads(Path(row["evaluation_artifacts"]["parent"]).read_text())
        assert baseline["adapter_hash"] == parent
        for draw in draws:
            assert draw["candidate_adapter_hash"] == row["candidate_adapter_hash"]
            assert draw["optimizer_steps"] == (0 if terminal else 3)
            item = json.loads(Path(draw["items_path"]).read_text())
            assert item["adapter_hash"] == draw["continuation_adapter_hash"]
            assert all(
                item["metrics"][h] - baseline["metrics"][h] == draw["h2"][h]
                for h in ("target", "ood", "retention")
            )
            assert all(len(item["items"][role]) == 2 for role in ROLES[1:])
            if terminal:
                assert (
                    draw["continuation_adapter_hash"] == row["candidate_adapter_hash"]
                )
                assert draw["items_path"] == row["evaluation_artifacts"]["t1"]
                assert draw["h2"] == row["labels"]["h1"]
        if not terminal:
            # Each original/fresh continuation starts from exactly this candidate's M1.
            # Distinct fixture batches can tokenize identically and yield equal
            # hashes; check each branch's five starts rather than hash counts.
            assert (
                starts[index * 7 + 2 : index * 7 + 7]
                == [row["candidate_adapter_hash"]] * 5
            )
    committed = backend.state_hash()
    assert committed in {r["candidate_adapter_hash"] for r in rows}
    assert accepted[0]["capture_item_outcomes"] is True
    assert accepted[0]["continuation_seed_offsets"] == [100, 200, 300, 400]
    # Exercise the real pilot inspector against the actual fixture artifacts.
    from forecasting_pilots import inspect_task

    sha = source_inventory()["source_tree_sha256"]
    write_json(tmp_path / "config.json", cfg.to_dict())
    write_json(
        tmp_path / "run.json",
        {
            "config_hash": cfg.fingerprint,
            "manifest_hash": digest(manifest),
            "source_tree_sha256": sha,
            "continuation_seed_offsets": [100, 200, 300, 400],
            "capture_item_outcomes": True,
        },
    )
    write_json(
        tmp_path / "completed.json",
        {"status": "complete", "completed_states": 1, "requested_generations": 1},
    )
    task = {
        "config": str(tmp_path / "config.json"),
        "manifest": str(tmp_path / "selection.json"),
        "output": str(tmp_path),
        "stream": "fixture/math",
        "seed": cfg.protocol.seed,
    }
    _, report = inspect_task(task, [100, 200, 300, 400], sha)
    assert report["terminal_draws"] == (15 if terminal else 0)
    item_file = Path(rows[0]["evaluation_artifacts"]["t1"])
    damaged = json.loads(item_file.read_text())
    damaged["items"]["target_dev"][0]["correct"] = 2
    write_json(item_file, damaged)
    with pytest.raises(ValueError, match="invalid per-item"):
        inspect_task(task, [100, 200, 300, 400], sha)


def test_item_capture_reuses_existing_predictions_and_rejects_mutation(tmp_path):
    from test_labeling import FixtureBackend
    from third_eye.config import Config

    backend = FixtureBackend(Config())
    splits = {r: [Example(r, "What is 1 plus 2?", "#### 3", r)] for r in ROLES[1:]}
    calls = []
    original = backend.generate

    def generate(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    backend.generate = generate
    plain = evaluate(backend, splits, VerifierRegistry(), 8, 42)
    calls.clear()
    recorded = evaluate(
        backend, splits, VerifierRegistry(), 8, 42, tmp_path / "items.json"
    )
    assert recorded == plain
    assert len(calls) == 3  # No second scoring pass.
    artifact = json.loads((tmp_path / "items.json").read_text())
    assert artifact["metrics"] == plain.to_dict()
    assert (
        artifact["items"]["target_dev"][0]["prompt_sha256"]
        == splits["target_dev"][0].prompt_hash
    )

    def mutate(*args, **kwargs):
        backend.state = (*backend.state, 1)
        return "#### 3"

    backend.generate = mutate
    with pytest.raises(RuntimeError, match="changed the adapter"):
        evaluate(backend, splits, VerifierRegistry(), 8, 42, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()
