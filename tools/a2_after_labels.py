"""Audit A2 labels, fit CPU forecasters and stop at Gate 2 review."""

from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]


def audit_labels(out, cfg, completed):
    from third_eye.data.schema import Correction, Example
    from third_eye.forecasting.dataset import read_records
    from third_eye.updates.candidates import batch_hash

    path = out / "meta_labels.jsonl"
    if not path.exists():
        if (
            completed["completed_states"] != 0
            or completed["status"] != "correction_scarcity"
        ):
            raise RuntimeError("Unexpected missing label artifact")
        return []
    rows = read_records([path])
    if len(rows) != 3 * completed["completed_states"]:
        raise RuntimeError("Label count differs from completed state count")
    compositions = {}
    for row in rows:
        if row["config_hash"] != cfg.fingerprint or row["protocol_amendment"] != "A2":
            raise RuntimeError("A2 label provenance mismatch")
        state = out / "states" / row["state_id"]
        branch = state / row["candidate_id"]
        root = json.loads((state / "correction_pool.json").read_text())
        continuation = json.loads((branch / "continuation_pool.json").read_text())
        for pool in [root, continuation]:
            stats = pool["stats"]
            ids = [c["example"]["id"] for c in pool["items"]]
            if (
                len(ids) != len(set(ids))
                or not stats["harvest_complete"]
                or stats["correction_attempt_limit"] != 8
                or stats["correction_seed_stride"] != 16
                or stats["verified_corrections"] != len(ids)
                or any(not 1 <= c["attempt"] <= 8 for c in pool["items"])
            ):
                raise RuntimeError("Invalid A2 verified correction pool")
        batch = json.loads((branch / "batch.json").read_text())
        ids = [c["example"]["id"] for c in batch]
        rootitems = {c["example"]["id"]: c for c in root["items"]}
        if (
            len(ids) != 2
            or len(set(ids)) != 2
            or any(c != rootitems.get(c["example"]["id"]) for c in batch)
        ):
            raise RuntimeError("Invalid unique candidate composition")
        if (
            batch_hash(
                [
                    Correction(Example(**c["example"]), c["completion"], c["attempt"])
                    for c in batch
                ]
            )
            != row["batch_hash"]
        ):
            raise RuntimeError("Candidate batch hash mismatch")
        compositions.setdefault(row["state_id"], set()).add(tuple(sorted(ids)))
        available = row["continuation_available"]
        if (
            available != int(len(continuation["items"]) >= 2)
            or row["runtime"]["candidate"]["optimizer_steps"] != 50
            or row["runtime"]["continuation"]["optimizer_steps"]
            != (50 if available else 0)
        ):
            raise RuntimeError("Optimizer budget or continuation availability mismatch")
        cbatch = json.loads((branch / "continuation_batch.json").read_text())
        if available:
            cids = [c["example"]["id"] for c in cbatch]
            citems = {c["example"]["id"]: c for c in continuation["items"]}
            if (
                len(cids) != 2
                or len(set(cids)) != 2
                or any(c != citems.get(c["example"]["id"]) for c in cbatch)
            ):
                raise RuntimeError("Invalid verified continuation batch")
        elif (
            cbatch
            or row["candidate_adapter_hash"] != row["continuation_adapter_hash"]
            or row["evaluation"]["t1"] != row["evaluation"]["t2"]
            or row["labels"]["h1"] != row["labels"]["h2"]
        ):
            raise RuntimeError("Terminal continuation changed M1")
        for head in ["target", "ood", "retention"]:
            for when in ["parent", "t1", "t2"]:
                value = row["evaluation"][when][head]
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise RuntimeError("Invalid evaluation metric")
            for horizon, when in [("h1", "t1"), ("h2", "t2")]:
                expected = (
                    row["evaluation"][when][head] - row["evaluation"]["parent"][head]
                )
                if not math.isclose(
                    row["labels"][horizon][head], expected, abs_tol=1e-12
                ):
                    raise RuntimeError("Consequence label mismatch")
    if any(len(compositions[state]) != 3 for state in compositions):
        raise RuntimeError("Candidate compositions are not distinct")
    return rows


def main():
    from third_eye.forecasting.dataset import read_records, split_records
    from third_eye.config import Config
    from third_eye.io import file_digest, write_json
    from third_eye.provenance import source_inventory
    from third_eye.statistics.report import phenomenon, gate1

    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Forecaster preparation requires CPU compute allocation")
    root = ROOT / "runs/a2"
    plan = json.loads((root / "execution_plan.json").read_text())
    if (
        source_inventory(ROOT)["source_tree_sha256"]
        != plan["resource_policy"]["source_tree_sha256"]
    ):
        raise RuntimeError("A2 source changed after freezing")
    paths, outcomes, by_stream = [], [], Counter()
    for task in plan["stages"]["labels"]:
        cmd = task["commands"][0]
        out = Path(cmd[cmd.index("--output") + 1])
        completed = json.loads((out / "completed.json").read_text())
        if completed["status"] not in {"complete", "correction_scarcity"}:
            raise RuntimeError("An A2 trajectory is incomplete")
        cfg = Config.load(cmd[cmd.index("--config") + 1])
        checked = audit_labels(out, cfg, completed)
        task_family = Path(cmd[cmd.index("--manifest") + 1]).parent.name
        by_stream[cfg.model.name + ":" + task_family] += len(checked) // 3
        if (out / "meta_labels.jsonl").exists():
            paths.append(str(out / "meta_labels.jsonl"))
        outcomes.append({"output": str(out), **completed})
    rows = read_records(paths)
    if not rows:
        write_json(
            root / "status.json",
            {
                "status": "insufficient_evidence",
                "reason": "All predeclared A2 trajectories were correction-starved",
                "trajectory_outcomes": outcomes,
            },
        )
        return
    if {r["protocol_amendment"] for r in rows} != {"A2"}:
        raise RuntimeError("Forecasters accept only A2 labels")
    stats = phenomenon(rows)
    decision = gate1(stats)
    reportdir = root / "study/phenomenon_a2"
    write_json(reportdir / "phenomenon.json", stats)
    write_json(
        reportdir / "gate1.json",
        {**decision, "evidence_hashes": {p: file_digest(p) for p in paths}},
    )
    parts = (
        split_records(rows, 42) if len({r["trajectory_id"] for r in rows}) >= 3 else {}
    )
    audit = {
        "trajectory_outcomes": outcomes,
        "states": len({r["state_id"] for r in rows}),
        "candidates": len(rows),
        "terminal_candidates": sum(r["continuation_available"] == 0 for r in rows),
        "states_by_model_task": dict(by_stream),
        "split_states": {k: len({r["state_id"] for r in v}) for k, v in parts.items()},
        "source_tree_sha256": plan["resource_policy"]["source_tree_sha256"],
        "protocol_amendment": "A2",
    }
    write_json(root / "label_audit.json", audit)
    if not decision["passed"]:
        write_json(
            root / "status.json",
            {
                "status": "a2_gate1_review_required",
                "gate1": decision,
                "forecasters_submitted": False,
            },
        )
        print("A2 phenomenon gate needs review; stopping before fits", flush=True)
        return
    # Empty root-starved trajectories remain logged but contain no usable labels.
    # Every fit receives the same available files and the same trajectory split.
    for index, task in enumerate(plan["stages"]["forecasters"]):
        command = task["commands"][0]
        start = command.index("--labels") + 1
        end = command.index("--output")
        command = [*command[:start], *paths, *command[end:]]
        subprocess.run([sys.executable, *command], check=True)
        print(f"Completed CPU forecaster {index + 1}/18", flush=True)
    for stage in ["forecast_reports", "analysis"]:
        for task in plan["stages"][stage]:
            for command in task["commands"]:
                if "--labels" in command:
                    start = command.index("--labels") + 1
                    end = command.index("--forecaster")
                    command = [*command[:start], *paths, *command[end:]]
                subprocess.run([sys.executable, *command], check=True)
    gate2 = json.loads((root / "study/analysis/gate2.json").read_text())
    write_json(
        root / "status.json",
        {
            "status": "gate2_review_required",
            "gate2": gate2,
            "online_submitted": False,
            "scaling_submitted": False,
        },
    )
    print(
        json.dumps(
            {"status": "gate2_review_required", "gate2_passed": gate2["passed"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
