"""Identify the actual working source and package versions, including edits."""

import importlib.metadata
from pathlib import Path

from third_eye.io import digest, file_digest


def source_inventory(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    paths = []
    for directory in ("src", "experiments"):
        paths.extend(
            p
            for p in (root / directory).rglob("*")
            if p.is_file() and p.suffix in {".py", ".slurm"}
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
