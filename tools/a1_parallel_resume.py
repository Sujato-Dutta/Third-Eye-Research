"""Resume the fixed A1 task grid with reviewed operational parallelism.

Scientific source and task argv stay unchanged. Existing GPU jobs are adopted,
never restarted. The replacement CPU controller stops at Gate 1 review.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


SEEDS = (1042, 2042, 3042, 4042, 5042)
STREAMS = {"qwen3_4b:math", "qwen3_4b:code", "llama3_2_3b:math", "llama3_2_3b:code"}


def pending_indices(tasks, submissions):
    expected = {(stream, seed) for stream in STREAMS for seed in SEEDS}
    if len(tasks) != 20 or {(t["stream"], t["seed"]) for t in tasks} != expected:
        raise ValueError("Require the unchanged twenty-state A1 grid")
    submitted = [i for row in submissions for i in row["indices"]]
    if len(submitted) != len(set(submitted)):
        raise ValueError("Task grid contains duplicate submissions")
    if any(type(i) is not int or not 0 <= i < len(tasks) for i in submitted):
        raise ValueError("Invalid submitted task index")
    return [i for i in range(len(tasks)) if i not in submitted]


def terminal_success(rows, expected):
    return len(rows) == expected and all(
        r[1] == "COMPLETED" and r[3] == "0:0" for r in rows
    )


def reservation(spent, submitted, pending, cap, available):
    if not 1 <= cap <= 20 or available <= 0:
        raise ValueError("Invalid reviewed allocation or parallelism")
    reserved = sum(
        row.get("reserved_hours", 0) for row in submitted if "billed_hours" not in row
    ) + 8 * len(pending)
    if spent + reserved > available:
        raise RuntimeError(
            "Fixed-task reservations exceed the reviewed resource envelope"
        )
    return reserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--old-controller", required=True)
    parser.add_argument("--max-parallel", type=int, default=20)
    parser.add_argument("--budget-hours", type=float, default=3000)
    parser.add_argument("--allocation-available", type=float, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Operational controller requires a Slurm compute allocation")
    root = args.root.resolve()
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "experiments"), str(root / "cluster")]
    from third_eye.config import Config
    from third_eye.io import file_digest, write_json
    from third_eye.provenance import source_inventory
    from third_eye.cluster import scheduler_job_id
    from third_eye.storage import check_storage
    from continue_study import accounting
    from a1_inspection import inspect_pilots, review_allows_calibration

    campaign = root / "runs/campaign_a1_20261003"
    operation = campaign / "parallel_resume_20261004"
    parent = json.loads((campaign / "campaign.json").read_text())
    parent_hash = file_digest(campaign / "campaign.json")
    plan_path = campaign / "plan.json"
    plan_hash = file_digest(plan_path)
    tasks = json.loads(plan_path.read_text())["stages"]["a1_pilots"]
    cpu = json.loads((root / "runs/a1_cpu_passed.json").read_text())
    gpu = json.loads((root / "runs/a1_gpu_passed.json").read_text())
    inventory = source_inventory(root)
    amendment_hash = file_digest(root / "docs/protocol_amendment_a1.md")
    if (
        cpu.get("status") != "passed"
        or gpu.get("status") != "passed"
        or any(
            proof["source_tree_sha256"] != inventory["source_tree_sha256"]
            for proof in (cpu, gpu, parent)
        )
    ):
        raise RuntimeError("Frozen source differs from validated scientific source")
    if (
        cpu["amendment_sha256"] != amendment_hash
        or parent["amendment_sha256"] != amendment_hash
    ):
        raise RuntimeError("Frozen A1 amendment changed")
    inspection_path = campaign / "pilot_inspection.json"
    inspection = inspect_pilots([t for t in tasks if t["seed"] == 1042])
    if inspection != json.loads(
        inspection_path.read_text()
    ) or not review_allows_calibration(
        inspection, inspection_path, parent.get("pilot_review_sha256")
    ):
        raise RuntimeError("Reviewed four-pilot evidence changed")
    unsubmitted = pending_indices(tasks, parent["submissions"])
    spent = parent["prior_gpu_hours"] + sum(
        s.get("billed_hours", 0) for s in parent["submissions"]
    )
    reserved = reservation(
        spent, parent["submissions"], unsubmitted, args.max_parallel, args.budget_hours
    )
    if args.budget_hours - spent + 100 > args.allocation_available:
        raise RuntimeError(
            "Resource envelope leaves insufficient allocation for CPU overhead"
        )
    active = [s for s in parent["submissions"] if "billed_hours" not in s]
    if sum(len(s["indices"]) for s in active) + len(unsubmitted) > args.max_parallel:
        raise RuntimeError("Overlapping adopted and new arrays exceed concurrency cap")
    deadline = datetime.fromisoformat(parent["completion_deadline_utc"]).timestamp()
    if time.time() + 8 * 3600 > deadline:
        raise RuntimeError("Remaining task reservation exceeds experimental deadline")
    storage = check_storage(
        Path(os.environ["THIRD_EYE_ORIGINAL_ROOT"]), 50, headroom_gb=8
    )
    for index, task in enumerate(tasks):
        command = task["commands"][0]
        config = Config.load(command[command.index("--config") + 1])
        if config.protocol.seed != task["seed"] or config.protocol.amendment != "A1":
            raise RuntimeError("Task configuration disagrees with frozen grid")
        if index in unsubmitted:
            status = (
                Path(json.loads(plan_path.read_text())["status_root"])
                / "a1_pilots"
                / f"{index}.json"
            )
            if status.exists() or Path(task["output"]).exists():
                raise RuntimeError("Unsubmitted task already has execution artifacts")
    operation.mkdir(exist_ok=True)
    proof_path = operation / "validation.json"
    validation = {
        "status": "passed",
        "validation_job": os.environ["SLURM_JOB_ID"],
        "parent_manifest_sha256": parent_hash,
        "plan_sha256": plan_hash,
        "source_tree_sha256": inventory["source_tree_sha256"],
        "operational_tool_sha256": file_digest(Path(__file__)),
        "operational_slurm_sha256": file_digest(
            root / "tools/a1_parallel_resume.slurm"
        ),
        "amendment_sha256": amendment_hash,
        "unsubmitted_indices": unsubmitted,
        "adopted_jobs": [s["job_id"] for s in active],
        "max_parallel": args.max_parallel,
        "global_budget_gpu_hours": args.budget_hours,
        "allocation_available_snapshot": args.allocation_available,
        "gpu_hours_spent": spent,
        "gpu_hours_reserved": reserved,
        "storage": storage,
        "scientific_protocol_changed": False,
    }
    if args.validate_only:
        if proof_path.exists():
            raise FileExistsError("Operational validation artifacts are immutable")
        write_json(proof_path, validation)
        print(json.dumps(validation), flush=True)
        return
    recorded = json.loads(proof_path.read_text())
    for key in (
        "parent_manifest_sha256",
        "plan_sha256",
        "source_tree_sha256",
        "operational_tool_sha256",
        "operational_slurm_sha256",
        "amendment_sha256",
        "unsubmitted_indices",
        "max_parallel",
        "global_budget_gpu_hours",
        "allocation_available_snapshot",
    ):
        if recorded[key] != validation[key]:
            raise RuntimeError(
                "Handoff differs from validated operational plan: " + key
            )
    old_rows = accounting(args.old_controller)
    if len(old_rows) != 1 or not old_rows[0][1].startswith("CANCELLED"):
        raise RuntimeError("Legacy CPU controller must be stopped before replacement")
    ledger_path = operation / "campaign.json"
    if ledger_path.exists():
        raise FileExistsError(
            "Replacement already launched; inspect ledger before any retry"
        )
    ledger = {
        **validation,
        "controller_job": os.environ["SLURM_JOB_ID"],
        "old_controller": args.old_controller,
        "submissions": parent["submissions"],
        "scientific_review_stops": ["gate1", "gate2"],
        "scaling_8b": "held",
    }
    write_json(operation / "original_campaign.json", parent)
    write_json(ledger_path, ledger)

    def event(status, **values):
        record = {
            "status": status,
            "utc": datetime.now(timezone.utc).isoformat(),
            "controller_job": os.environ["SLURM_JOB_ID"],
            "operational_ledger": str(ledger_path),
            "max_parallel": args.max_parallel,
            "scaling_8b": "held",
            **values,
        }
        write_json(campaign / "status.json", record)
        write_json(operation / "status.json", record)
        print(json.dumps(record), flush=True)

    if unsubmitted:
        claim = operation / "submission.claimed"
        with claim.open("x") as handle:
            json.dump({"indices": unsubmitted, "plan_sha256": plan_hash}, handle)
        env = dict(
            os.environ, THIRD_EYE_PLAN=str(plan_path), THIRD_EYE_STAGE="a1_pilots"
        )
        job = scheduler_job_id(
            subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    "--partition=gh",
                    "--account=IRI23021",
                    "--array="
                    + ",".join(map(str, unsubmitted))
                    + "%"
                    + str(args.max_parallel),
                    "--time=08:00:00",
                    "experiments/jobs/vista_study.slurm",
                ],
                env=env,
                text=True,
            )
        )
        row = {
            "job_id": job,
            "indices": unsubmitted,
            "reserved_hours": len(unsubmitted) * 8,
            "phase": "parallel_remaining",
        }
        ledger["submissions"].append(row)
        active.append(row)
        write_json(ledger_path, ledger)
    event("waiting_for_parallel_calibrations", jobs=[s["job_id"] for s in active])
    while active:
        for row in list(active):
            query = subprocess.run(
                ["squeue", "-h", "-j", row["job_id"], "-o", "%i %T"],
                text=True,
                capture_output=True,
            )
            if query.returncode and "Invalid job id" not in query.stderr:
                raise RuntimeError("Scheduler query failed: " + query.stderr)
            if query.stdout.strip():
                continue
            records = accounting(row["job_id"])
            if not terminal_success(records, len(row["indices"])):
                event(
                    "stopped",
                    reason="Array missing or not fully successful",
                    accounting=records,
                )
                raise RuntimeError("Parallel calibration array incomplete or failed")
            row["billed_hours"] = sum(max(0.25, int(r[2]) / 3600) for r in records)
            write_json(ledger_path, ledger)
            active.remove(row)
        if active:
            time.sleep(120)
    labels = [
        str(Path(t["output"]) / "meta_labels.jsonl")
        for t in tasks
        if (Path(t["output"]) / "meta_labels.jsonl").exists()
    ]
    report = campaign / "gate1"
    if labels and not report.exists():
        subprocess.run(
            [
                sys.executable,
                "experiments/analyze.py",
                "--labels",
                *labels,
                "--output",
                str(report),
            ],
            check=True,
        )
    evidence = json.loads((report / "gate1.json").read_text())
    event(
        "gate1_review_required",
        gate1=evidence,
        note="No automatic forecasting, transfer, or 8B launch",
    )
    target = Path(os.environ["WORK"]) / "third_eye_results/campaign_a1_20261003"
    for name in ("runs", "tools", "docs"):
        destination = target / name
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", str(root / name) + "/", str(destination) + "/"], check=True
        )


if __name__ == "__main__":
    main()
