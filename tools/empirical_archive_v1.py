# ruff: noqa: E402
"""Durable compute-node archive of the initial supplement and preserved failures."""

import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_common_v1 import BASE, checked, prepared
from third_eye.io import file_digest, write_json


def main():
    checked()
    paths = set()
    for p in BASE.rglob("*"):
        if "comparator_venv" in p.parts:
            continue
        if p.is_symlink():
            raise RuntimeError("Unexpected artifact symlink")
        if p.is_file():
            paths.add(p)
    for task in prepared()["noise_tasks"]:
        paths.update(Path(p) for p in task["input_sha256"])
    for pattern in ["empirical_*.py", "empirical_*.slurm", "test_empirical_*.py"]:
        paths.update((ROOT / "tools").glob(pattern))
    paths.update((ROOT / "docs").glob("protocol_empirical*.md"))
    paths.update(ROOT.glob("third_eye_empirical_*.out"))
    target = Path(
        "/work/11617/sujato_ts/vista/third_eye_results/empirical_v1_initial_20261008.tar.gz"
    )
    if target.exists():
        raise RuntimeError("Durable artifact exists; never overwrite it")
    inventory = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(paths)}
    record = BASE / "initial_archive_manifest.json"
    write_json(
        record,
        dict(
            status="initial_supplement_and_failed_validation_evidence",
            files=inventory,
            noise_measurements_released=False,
            replication_released=False,
        ),
    )
    with tarfile.open(target, "w:gz") as archive:
        for p in sorted(paths | {record}):
            archive.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
    write_json(
        BASE / "initial_archive_receipt.json",
        dict(path=str(target), sha256=file_digest(target), bytes=target.stat().st_size),
    )
    print(
        json.dumps(dict(archive=str(target), bytes=target.stat().st_size)), flush=True
    )


if __name__ == "__main__":
    main()
