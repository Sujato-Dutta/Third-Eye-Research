"""Publish GPU preflight only after its CPU/source and CUDA checks pass."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.cluster import require_gpu_allocation
from third_eye.io import file_digest, write_json
from third_eye.provenance import execution_environment, source_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-job", required=True)
    args = parser.parse_args()
    require_gpu_allocation()
    job = os.environ["SLURM_JOB_ID"]
    parent = Path("runs/vista_preflight") / args.cpu_job / "passed.json"
    proof = json.loads(parent.read_text())
    inventory = source_inventory()
    if (
        proof.get("status") != "passed"
        or proof["source_tree_sha256"] != inventory["source_tree_sha256"]
    ):
        raise RuntimeError("CPU preflight failed or project source changed")
    for path in (
        Path(f"runs/vista_backend_{job}.json"),
        Path(f"runs/vista_gpu_profile_{job}.json"),
    ):
        if json.loads(path.read_text()).get("status") != "passed":
            raise RuntimeError("CUDA preflight has not passed")
    output = Path("runs/vista_preflight") / job / "passed.json"
    if output.exists():
        raise FileExistsError("GPU preflight evidence is immutable")
    write_json(
        output,
        {
            **proof,
            "cpu_setup_job": args.cpu_job,
            "cpu_setup_proof_sha256": file_digest(parent),
            "execution": execution_environment(),
            **inventory,
        },
    )


if __name__ == "__main__":
    main()
