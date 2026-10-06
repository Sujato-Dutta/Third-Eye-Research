"""Fail closed unless two allocated nodes reproduce A2 control artifacts."""

import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from third_eye.io import digest, file_digest, write_json
    from third_eye.provenance import source_inventory

    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Review requires a scheduled CPU allocation")
    root = ROOT / "runs/a2/validation/v2"
    records = [json.loads((root / f"cuda_{i}.json").read_text()) for i in range(2)]
    cpu = json.loads((ROOT / "runs/a2/validation/cpu.json").read_text())
    validation = json.loads((root / "cpu_v2.json").read_text())
    if not validation["passed"] or any(
        file_digest(ROOT / p) != h
        for p, h in validation["validation_files_sha256"].items()
    ):
        raise RuntimeError("V2 validation tools changed after CPU checks")
    current = source_inventory(ROOT)["source_tree_sha256"]
    plan = json.loads((ROOT / "runs/a2/execution_plan.json").read_text())
    if any(
        file_digest(ROOT / p) != h
        for p, h in plan["resource_policy"]["operational_files_sha256"].items()
    ):
        raise RuntimeError("Operational files changed after validation")
    if (
        cpu["source_tree_sha256"] != current
        or file_digest(ROOT / "runs/a2/execution_plan.json") != cpu["plan_sha256"]
    ):
        raise RuntimeError("Validated source/plan changed")
    for r in records:
        if (
            not r["passed"]
            or r["source_tree_sha256"] != current
            or r["comparable_sha256"] != digest(r["models"])
        ):
            raise RuntimeError("Invalid CUDA verification proof")
    hosts = [json.loads((root / f"step_host_{i}.json").read_text()) for i in range(2)]
    if len({h["host"] for h in hosts}) != 2 or any(
        h["slurm_job_id"] != r["execution"]["SLURM_JOB_ID"] or h["rank"] != i
        for i, (h, r) in enumerate(zip(hosts, records))
    ):
        raise RuntimeError(
            "Cross-node verification needs two distinct allocated compute hosts"
        )
    if (
        records[0]["models"] != records[1]["models"]
        or records[0]["software_versions"] != records[1]["software_versions"]
    ):
        write_json(
            root / "gpu_review_failure.json",
            {
                "status": "cross_node_difference",
                "passed": False,
                "control_hashes": [r["comparable_sha256"] for r in records],
            },
        )
        raise RuntimeError("Cross-node A2 reproducibility did not pass")
    write_json(
        ROOT / "runs/a2/validation/gpu_review.json",
        {
            "status": "passed",
            "passed": True,
            "source_tree_sha256": current,
            "control_sha256": records[0]["comparable_sha256"],
            "nodes": [h["host"] for h in hosts],
            "allocated_node_lists": [
                r["execution"]["SLURM_JOB_NODELIST"] for r in records
            ],
            "step_hosts": hosts,
            "evidence_sha256": {
                str(root / f"cuda_{i}.json"): file_digest(root / f"cuda_{i}.json")
                for i in range(2)
            },
            "reviewer_sha256": file_digest(Path(__file__)),
            "development_launcher_sha256": file_digest(
                ROOT / "tools/a2_cuda_verify_dev_v2.slurm"
            ),
            "historical_a1_root_cause": "not established; A2 controls are separately tested",
            "scope": "tested fixed GH200 environment and control prompts",
            "validation_revision": "V2: nullable direct-source metadata",
            "cpu_v2_sha256": file_digest(root / "cpu_v2.json"),
        },
    )
    print(
        "PASS: cross-node A2 CUDA, decoding, verifier and fifty-step adapter controls",
        flush=True,
    )


if __name__ == "__main__":
    main()
