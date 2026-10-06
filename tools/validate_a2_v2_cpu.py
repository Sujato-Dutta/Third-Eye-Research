"""Bind the metadata-logging fix to the unchanged A2 plan and science source."""

import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from third_eye.io import file_digest, write_json
    from third_eye.provenance import source_inventory

    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("V2 validation requires scheduled CPU compute")
    cpu = json.loads((ROOT / "runs/a2/validation/cpu.json").read_text())
    plan = json.loads((ROOT / "runs/a2/execution_plan.json").read_text())
    if (
        not cpu["passed"]
        or cpu["plan_sha256"] != file_digest(ROOT / "runs/a2/execution_plan.json")
        or cpu["source_tree_sha256"] != source_inventory(ROOT)["source_tree_sha256"]
    ):
        raise RuntimeError("Frozen A2 scientific source or plan changed")
    if any(
        file_digest(ROOT / p) != h
        for p, h in plan["resource_policy"]["operational_files_sha256"].items()
    ):
        raise RuntimeError("Archived validated operational files changed")
    names = [
        "tools/validation_source_identity.py",
        "tools/test_validation_source_identity.py",
        "tools/verify_a2_cuda_v2.py",
        "tools/review_a2_cuda_dev_v2.py",
        "tools/validate_a2_v2_cpu.py",
        "tools/a2_cpu_verify_v2.slurm",
        "tools/a2_cuda_verify_dev_v2.slurm",
        "tools/a2_cuda_review_dev_v2.slurm",
    ]
    write_json(
        ROOT / "runs/a2/validation/v2/cpu_v2.json",
        {
            "status": "passed",
            "passed": True,
            "job_id": os.environ["SLURM_JOB_ID"],
            "source_tree_sha256": cpu["source_tree_sha256"],
            "plan_sha256": cpu["plan_sha256"],
            "fix": "Direct pinned model sources may legitimately have no mirror proof; preserve per-model diagnostics",
            "validation_files_sha256": {p: file_digest(ROOT / p) for p in names},
            "scientific_protocol_unchanged": True,
        },
    )
    print(
        "PASS: V2 metadata validation bound to unchanged A2 source and plan", flush=True
    )


if __name__ == "__main__":
    main()
