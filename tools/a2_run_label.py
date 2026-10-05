"""Check frozen A2 source and operational files before executing a GPU task."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from third_eye.cluster import require_gpu_allocation
    from third_eye.io import file_digest
    from third_eye.provenance import source_inventory

    require_gpu_allocation()
    plan_path = ROOT / "runs/a2/execution_plan.json"
    plan = json.loads(plan_path.read_text())
    policy = plan["resource_policy"]
    if source_inventory(ROOT)["source_tree_sha256"] != policy["source_tree_sha256"]:
        raise RuntimeError("Frozen A2 scientific source changed")
    if any(
        file_digest(ROOT / p) != h
        for p, h in policy["operational_files_sha256"].items()
    ):
        raise RuntimeError("Frozen A2 operational files changed")
    subprocess.run(
        [
            sys.executable,
            "experiments/task.py",
            "--plan",
            str(plan_path),
            "--stage",
            "labels",
            "--index",
            os.environ["SLURM_ARRAY_TASK_ID"],
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
