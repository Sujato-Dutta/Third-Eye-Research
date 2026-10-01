"""Run one structured study task within a SLURM compute allocation."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.cluster import require_gpu_allocation
from third_eye.io import write_json, file_digest
from third_eye.storage import check_storage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--index", type=int, default=0)
    a = p.parse_args()
    require_gpu_allocation()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    tasks = plan["stages"].get(a.stage)
    if tasks is None or not 0 <= a.index < len(tasks):
        p.error("Unknown stage/task index")
    task = tasks[a.index]
    storage = check_storage(Path.cwd(), plan.get("storage_quota_gb", 50))
    for gate in task.get("requires_gates", []):
        evidence = json.loads(Path(gate).read_text(encoding="utf-8"))
        if evidence.get("passed") is not True:
            p.error("Required research gate has not passed: " + gate)
    status = Path(plan["status_root"]) / a.stage / f"{a.index}.json"
    if status.exists():
        p.error(
            "Task already has a status artifact; review failures and create a versioned rerun plan"
        )
    started = time.time()
    common = {
        "stage": a.stage,
        "index": a.index,
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "plan_sha256": file_digest(a.plan),
        "commands": task["commands"],
        "storage_preflight": storage,
    }
    write_json(status, {**common, "status": "running"})
    try:
        for command in task["commands"]:
            # Command argv is explicit; no arbitrary shell interpolation.
            subprocess.run([sys.executable, *command], check=True)
        write_json(
            status, {**common, "status": "complete", "seconds": time.time() - started}
        )
    except Exception as exc:
        write_json(
            status,
            {
                **common,
                "status": "failed",
                "seconds": time.time() - started,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
