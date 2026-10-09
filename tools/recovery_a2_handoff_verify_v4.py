# ruff: noqa: E402
"""Verify a completely prepared V4 graph after the obsolete-job cleanup failed."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from recovery_a2_controller_v4 import frozen, REC
from third_eye.io import file_digest, write_json


def state(job):
    return subprocess.check_output(
        ["sacct", "-j", str(job), "-X", "-n", "-P", "--format=State,ExitCode"],
        text=True,
    ).strip()


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Reconciliation validation requires compute allocation")
    frozen()
    if state("1057520") != "COMPLETED|0:0" or state("1057521") != "COMPLETED|0:0":
        raise RuntimeError("CPU/CUDA science validation must pass")
    if state("1057522") != "FAILED|1:0":
        raise RuntimeError("Unexpected controller state")
    log = (ROOT / "third_eye_a2_recovery_cpu_1057522.out").read_text()
    if (
        "Command '['squeue', '-h', '-j', '1049535', '-o', '%T']' returned non-zero exit status 1"
        not in log
    ):
        raise RuntimeError(
            "Failure does not match obsolete-job cleanup; do not change dependencies"
        )
    passed = json.loads((REC / "cuda_passed.json").read_text())
    if (
        passed["status"] != "passed"
        or passed["release_sha256"] != file_digest(REC / "release_v4.json")
        or sorted(c["index"] for c in passed["cases"]) != [20, 22, 24]
    ):
        raise RuntimeError("Invalid CUDA checkpoint proof")
    handoff = json.loads((REC / "handoff.json").read_text())
    if handoff["timeout_indices"] != [20, 22, 24] or handoff[
        "derivative_plan_sha256"
    ] != file_digest(REC / "execution_plan.json"):
        raise RuntimeError("Incomplete derivative plan")
    jobs = []
    for index, case in zip(
        [20, 22, 24], sorted(passed["cases"], key=lambda c: c["index"])
    ):
        path = REC / f"task_{index}.json"
        context = json.loads(path.read_text())
        receipt = json.loads((REC / f"submission_{index}.json").read_text())
        if file_digest(path) != receipt["manifest_sha256"] or any(
            file_digest(p) != h for p, h in context["input_sha256"].items()
        ):
            raise RuntimeError("Prepared input changed")
        partial = next(b for b in context["branches"] if not b["complete"])
        if (
            case["parent_hash"] != context["parent_adapter_hash"]
            or case["partial_t1_hash"] != partial["t1_hash"]
            or not all(
                case[k]
                for k in [
                    "checkpoint_matches",
                    "rollback_exact",
                    "no_new_optimizer_steps",
                ]
            )
        ):
            raise RuntimeError("Checkpoint proof differs from prepared task")
        jobs.append(receipt["job_id"])
    if jobs != handoff["recovery_gpu_jobs"]:
        raise RuntimeError("Handoff jobs differ")
    all_jobs = jobs + [handoff["final_cpu_job"]]
    for job in all_jobs:
        if (
            subprocess.check_output(
                ["squeue", "-h", "-j", job, "-o", "%T"], text=True
            ).strip()
            != "PENDING"
        ):
            raise RuntimeError("Graph changed state; review before reconciliation")
    target = REC / "handoff_reconciliation_v1.json"
    if target.exists():
        raise RuntimeError("Reconciliation proof already exists")
    write_json(
        target,
        dict(
            status="validated_scheduler_handoff_repair",
            old_controller_job="1057522",
            cpu_validation_job="1057520",
            cuda_validation_job="1057521",
            new_anchor_job=os.environ["SLURM_JOB_ID"],
            recovery_gpu_jobs=jobs,
            final_cpu_job=handoff["final_cpu_job"],
            release_sha256=file_digest(REC / "release_v4.json"),
            scientific_validation_bypassed=False,
            reason="Controller failed only after complete preparation, when querying an obsolete purged job",
        ),
    )


if __name__ == "__main__":
    main()
