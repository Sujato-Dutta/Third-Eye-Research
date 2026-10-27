"""Allocated, allowlisted cleanup of recoverable downloads; preserve research."""

import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def usage(home):
    result = subprocess.run(
        ["du", "-x", "-B1", "-s", str(home)],
        text=True,
        capture_output=True,
        check=True,
    )
    return int(result.stdout.split()[0])


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Cleanup and storage scans require a CPU Slurm allocation")
    home = Path.home().resolve()
    root = home / "third_eye_dgx_20261026"
    root.mkdir(exist_ok=True)
    dest = root / "cleanup_receipt"
    dest.mkdir(exist_ok=False)
    targets = [
        home / "MINDSET_v2/.cache/gpu_prototype/huggingface/hub/models--Qwen--Qwen3.5-9B",
        home / "ollama-linux-amd64-0.34.0.tar.zst",
    ]
    for target in targets:
        target.resolve().relative_to(home)
        if target.is_symlink() or target.resolve() == home:
            raise RuntimeError("Refuse symlink/root cleanup target")
        if target.exists() and target.stat().st_uid != os.getuid():
            raise RuntimeError("Cleanup target is not owned by this account")
    active = subprocess.run(
        ["squeue", "-h", "-u", str(os.getuid()), "-o", "%j"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if "mindset" in active.lower():
        raise RuntimeError("Another project may be using its download cache")
    record = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "policy_limit_bytes": 50_000_000_000,
        "before_bytes": usage(home),
        "targets": [],
        "research_results_and_environments_preserved": True,
    }
    for index, target in enumerate(targets):
        files = []
        if target.is_dir():
            for path in sorted(target.rglob("*")):
                if path.is_file() or path.is_symlink():
                    item = {
                        "path": str(path.relative_to(home)),
                        "bytes": path.stat().st_size,
                        "symlink": os.readlink(path) if path.is_symlink() else None,
                    }
                    files.append(item)
                    # Keep cached configuration/tokenizer/provenance metadata.
                    if path.stat().st_size < 32 * 1024**2 and path.name.endswith(
                        (".json", ".jinja", ".txt", ".md")
                    ):
                        copy = dest / "metadata" / str(index) / path.relative_to(target)
                        copy.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, copy)
        elif target.is_file():
            files.append({"path": str(target.relative_to(home)), "bytes": target.stat().st_size})
        record["targets"].append({"path": str(target), "inventory": files})
    receipt = dest / "receipt.json"
    receipt.write_text(json.dumps(record, indent=2))
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    record["after_bytes"] = usage(home)
    record["reclaimed_bytes"] = record["before_bytes"] - record["after_bytes"]
    record["completed"] = True
    receipt.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "targets"}))


if __name__ == "__main__":
    main()
