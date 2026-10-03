"""Identify the actual working source and package versions, including edits."""

import importlib.metadata
import os
from pathlib import Path

from third_eye.io import digest, file_digest


def source_inventory(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    paths = []
    for directory in ("src", "experiments", "cluster"):
        paths.extend(
            p
            for p in (root / directory).rglob("*")
            if p.is_file() and p.suffix in {".py", ".slurm", ".sh"}
        )
    paths.extend(root.glob("requirements*.txt"))
    files = {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(paths)}
    return {"source_tree_sha256": digest(files), "source_files": files}


def versions():
    result = {}
    for package in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "datasets",
        "math-verify",
        "bitsandbytes",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def execution_environment():
    """Allocated resources and verifier settings, excluding credentials."""
    result = {
        name: os.environ.get(name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_GPUS",
            "SLURM_CLUSTER_NAME",
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_PARTITION",
            "THIRD_EYE_VERIFIER_WORKERS",
            "THIRD_EYE_SYMBOLIC_WORKERS",
        )
    }
    sandbox = os.environ.get("THIRD_EYE_SANDBOX_CONFIG")
    if sandbox:
        result["sandbox_config_sha256"] = file_digest(sandbox)
    import torch

    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["gpu"] = {
            "name": device.name,
            "total_memory_bytes": device.total_memory,
            "multiprocessors": device.multi_processor_count,
        }
    return result
