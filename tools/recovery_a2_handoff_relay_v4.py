"""Bounded scheduler-only reconciliation; no research imports or computation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--deadline", required=True)
    a = p.parse_args()
    import os
    import socket

    if "login" not in socket.gethostname().lower() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scheduler relay requires login host")
    rec = a.root / "runs/a2/recovery_20261008_v4"
    deadline = datetime.fromisoformat(a.deadline)
    while datetime.now(timezone.utc) < deadline:
        state = subprocess.check_output(
            [
                "/usr/bin/sacct",
                "-j",
                a.anchor,
                "-X",
                "-n",
                "-P",
                "--format=State,ExitCode",
            ],
            text=True,
        ).strip()
        if state == "COMPLETED|0:0":
            break
        if state and state.split("|")[0] not in ["PENDING", "RUNNING", "COMPLETING"]:
            raise RuntimeError("Reconciliation validation failed: " + state)
        time.sleep(5)
    else:
        raise RuntimeError("Bounded reconciliation deadline passed")
    proof = json.loads((rec / "handoff_reconciliation_v1.json").read_text())
    if (
        proof["status"] != "validated_scheduler_handoff_repair"
        or proof["new_anchor_job"] != a.anchor
        or proof["scientific_validation_bypassed"]
    ):
        raise RuntimeError("Unexpected reconciliation proof")
    if (
        hashlib.sha256((rec / "release_v4.json").read_bytes()).hexdigest()
        != proof["release_sha256"]
    ):
        raise RuntimeError("V4 release changed")
    if (
        proof["recovery_gpu_jobs"] != ["1057528", "1057529", "1057530"]
        or proof["final_cpu_job"] != "1057531"
    ):
        raise RuntimeError("Unexpected queue entries")
    receipt = rec / "handoff_reconciled_v1.json"
    if receipt.exists():
        raise RuntimeError("Reconciliation already attempted; inspect it")
    record = dict(status="updating", anchor=a.anchor, updated=[])
    receipt.write_text(json.dumps(record, indent=2))
    for job in proof["recovery_gpu_jobs"] + [proof["final_cpu_job"]]:
        if (
            subprocess.check_output(
                ["/usr/bin/squeue", "-h", "-j", job, "-o", "%T"], text=True
            ).strip()
            != "PENDING"
        ):
            raise RuntimeError("Job changed state before update")
        dependencies = (
            [a.anchor]
            if job != proof["final_cpu_job"]
            else [a.anchor, *proof["recovery_gpu_jobs"]]
        )
        subprocess.run(
            [
                "/usr/bin/scontrol",
                "update",
                "JobId=" + job,
                "Dependency=afterok:" + ":".join(dependencies),
            ],
            check=True,
        )
        record["updated"].append(
            dict(job_id=job, dependency="afterok:" + ":".join(dependencies))
        )
        receipt.write_text(json.dumps(record, indent=2))
    record["status"] = "reconciled"
    receipt.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
