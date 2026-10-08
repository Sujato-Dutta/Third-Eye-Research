"""Provisional forecasting on immutable published states; no science gate release."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from a2_after_labels import audit_labels
from third_eye.config import Config
from third_eye.forecasting.dataset import read_records, split_records
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.statistics.report import phenomenon, gate1

BASE = ROOT / "runs/a2"
OUT = BASE / "provisional_forecasting_20261008"


def published_rows(rows, ledger):
    states = [r for r in ledger if r.get("status") == "complete" and "candidates" in r]
    assert states and all(r["candidates"] == 3 for r in states)
    ids = [r["state_id"] for r in states]
    assert len(ids) == len(set(ids))
    selected = [r for r in rows if r["state_id"] in set(ids)]
    counts = Counter(r["state_id"] for r in selected)
    assert set(counts) == set(ids) and all(n == 3 for n in counts.values())
    assert all(
        len({r["candidate_id"] for r in selected if r["state_id"] == sid}) == 3
        for sid in ids
    )
    return selected


def assignments(rows):
    parts = split_records(rows, 42)
    return {p: sorted({r["trajectory_id"] for r in rs}) for p, rs in parts.items()}


def remap(command, paths, original, output):
    original, output = original.as_posix(), output.as_posix()
    result = [
        str(output) + c[len(str(original)) :]
        if c.startswith(str(original) + "/")
        else c
        for c in command
    ]
    if "--labels" in result:
        start = result.index("--labels") + 1
        end = (
            result.index("--output")
            if "--forecaster" not in result
            else result.index("--forecaster")
        )
        result = [*result[:start], *paths, *result[end:]]
    if "--device" in result:
        assert result[result.index("--device") + 1] == "cpu"
    if "--split-seed" in result:
        assert result[result.index("--split-seed") + 1] == "42"
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Provisional forecasting requires Slurm compute allocation")
    plan = json.loads((BASE / "execution_plan.json").read_text())
    policy = plan["resource_policy"]
    assert source_inventory(ROOT)["source_tree_sha256"] == policy["source_tree_sha256"]
    assert all(
        file_digest(ROOT / p) == h
        for p, h in policy["operational_files_sha256"].items()
    )
    release = json.loads((BASE / "prefix_forecast_release_20261008.json").read_text())
    assert all(file_digest(ROOT / p) == h for p, h in release["files"].items())
    assert not OUT.exists(), (
        "Provisional outputs are immutable; inspect any previous attempt"
    )
    OUT.mkdir(parents=True)
    paths, outcomes, trajectory_ids = [], [], set()
    for index, task in enumerate(plan["stages"]["labels"]):
        cmd = task["commands"][0]
        original = Path(cmd[cmd.index("--output") + 1])
        cfg = Config.load(cmd[cmd.index("--config") + 1])
        run_bytes = (original / "run.json").read_bytes()
        run = json.loads(run_bytes)
        for attempt in range(3):
            try:
                ledger = [
                    json.loads(line)
                    for line in (original / "ledger.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                rows = [
                    json.loads(line)
                    for line in (original / "meta_labels.jsonl")
                    .read_text()
                    .splitlines()
                    if line.strip()
                ]
                break
            except json.JSONDecodeError:
                if attempt == 2:
                    raise
                time.sleep(0.25)
        rows = published_rows(rows, ledger)
        assert all(r["trajectory_id"] == run["trajectory_id"] for r in rows)
        assert all(
            r["seed"] == cfg.protocol.seed
            and r["model"] == cfg.model.name
            and r["manifest_hash"] == run["manifest_hash"]
            and r["state_id"]
            == digest(
                {
                    "parent": r["parent_adapter_hash"],
                    "config": cfg.fingerprint,
                    "splits": run["manifest_hash"],
                    "generation": r["generation"],
                }
            )[:20]
            for r in rows
        )
        assert run["trajectory_id"] not in trajectory_ids
        trajectory_ids.add(run["trajectory_id"])
        view = OUT / "snapshot" / f"task_{index}"
        (view / "states").mkdir(parents=True)
        for sid in {r["state_id"] for r in rows}:
            (view / "states" / sid).symlink_to(
                original / "states" / sid, target_is_directory=True
            )
        (view / "run.json").write_bytes(run_bytes)
        (view / "meta_labels.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        )
        checked = audit_labels(
            view,
            cfg,
            {"status": "published_prefix", "completed_states": len(rows) // 3},
        )
        assert len(checked) == len(rows)
        paths.append(str(view / "meta_labels.jsonl"))
        outcomes.append(
            {
                "index": index,
                "source": str(original),
                "trajectory_id": run["trajectory_id"],
                "published_states": len(rows) // 3,
                "source_run_sha256": file_digest(original / "run.json"),
                "snapshot_labels_sha256": file_digest(view / "meta_labels.jsonl"),
                "published_record_hashes": [digest(r) for r in rows],
                "complete_trajectory": len(rows) == 15,
            }
        )
    assert len(trajectory_ids) == len(plan["stages"]["labels"]) == 36
    rows = read_records(paths)
    parts = split_records(rows, 42)
    assert {r["trajectory_id"] for rs in parts.values() for r in rs} == trajectory_ids
    partition = assignments(rows)
    write_json(
        OUT / "snapshot_manifest.json",
        {
            "provisional": True,
            "final_gate2_authorized": False,
            "original_plan_sha256": file_digest(BASE / "execution_plan.json"),
            "release_sha256": file_digest(
                BASE / "prefix_forecast_release_20261008.json"
            ),
            "source_tree_sha256": policy["source_tree_sha256"],
            "job_id": os.environ["SLURM_JOB_ID"],
            "states": len(rows) // 3,
            "candidates": len(rows),
            "trajectories": outcomes,
            "trajectory_assignments": partition,
            "partition_states": {
                p: len({r["state_id"] for r in rs}) for p, rs in parts.items()
            },
        },
    )
    decision = gate1(phenomenon(rows))
    write_json(OUT / "phenomenon_gate1.json", {**decision, "provisional": True})
    if not decision["passed"]:
        write_json(
            OUT / "status.json",
            {
                "status": "provisional_phenomenon_review_required",
                "online_submitted": False,
            },
        )
        return
    study = OUT / "study"
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) >= 8
    (OUT / "fit_logs").mkdir()

    def fit(i, task):
        command = remap(task["commands"][0], paths, BASE / "study", study)
        with (OUT / "fit_logs" / f"fit_{i:02d}.log").open("w") as log:
            subprocess.run(
                [sys.executable, *command],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return i

    # Each independent process retains the frozen four-thread workload setting.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(fit, i, task)
            for i, task in enumerate(plan["stages"]["forecasters"])
        ]
        for future in as_completed(futures):
            i = future.result()
            print(f"Provisional fit {i + 1}/{len(futures)} complete", flush=True)
    for stage in ["forecast_reports", "analysis"]:
        for task in plan["stages"][stage]:
            for command in task["commands"]:
                subprocess.run(
                    [sys.executable, *remap(command, paths, BASE / "study", study)],
                    check=True,
                )
    measured = json.loads((study / "analysis/gate2.json").read_text())
    write_json(
        OUT / "status.json",
        {
            "status": "provisional_forecasting_review_required",
            "provisional": True,
            "provisional_gate2": measured,
            "final_gate2_authorized": False,
            "online_submitted": False,
            "scaling_submitted": False,
            "confirmation_required": "Full recovered dataset with identical trajectory assignments and frozen fit settings",
        },
    )
    print(
        json.dumps(
            {
                "status": "provisional_forecasting_review_required",
                "states": len(rows) // 3,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
