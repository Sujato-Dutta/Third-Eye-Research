"""Scheduler-only E2 handoff: queue collection behind both validation jobs.

This tool only checks small release metadata and calls Slurm. It imports no
research, training or analysis code. Runtime scientific checks remain unchanged.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def scheduler(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def state(job):
    return scheduler(["/usr/bin/squeue", "-h", "-j", job, "-o", "%T"]).strip()


def gpu_nodes():
    raw = scheduler(["/usr/bin/squeue", "-h", "-u", "sujato_ts", "-o", "%P|%D"])
    return sum(
        int(line.split("|")[1])
        for line in raw.splitlines()
        if set(line.split("|")[0].split(",")) & {"gh", "gh-dev"}
    )


def main():
    if "login" not in socket.gethostname().lower() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scheduler-only handoff requires login node")
    root = Path.cwd()
    out = root / "runs/a2/empirical_v1/checkpoint_fallback_v3"
    release = json.loads((out / "release.json").read_text())
    if any(
        sha(root / name) != value for name, value in release["files"].items()
    ) or any(sha(Path(p)) != h for p, h in release["prerequisites"].items()):
        raise RuntimeError("Frozen E2 release or prerequisites changed")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(
        release["scope"]["deadline_utc"]
    ):
        raise RuntimeError("Deadline passed")
    chain = json.loads((out / "validation_chain.json").read_text())["jobs"]
    cpu, cuda, old_controller = (
        chain[k] for k in ("cpu_validation", "cuda_validation", "launch_controller")
    )
    if state(old_controller) != "PENDING":
        raise RuntimeError(
            "Original launch controller changed state; inspect before handoff"
        )
    if state(cpu) not in {"PENDING", "RUNNING"} or state(cuda) not in {
        "PENDING",
        "RUNNING",
    }:
        raise RuntimeError("Validation state changed; retain original handoff")
    tasks = json.loads((root / "runs/a2/empirical_v1/prepared.json").read_text())[
        "replication_tasks"
    ]
    if len(tasks) != 8 or any(Path(t["output"]).exists() for t in tasks):
        raise RuntimeError("Need exactly eight untouched fresh outputs")
    target = out / "trajectory_submissions.json"
    if target.exists():
        raise RuntimeError("Previous submission attempt exists; no blind retry")
    record = dict(
        status="submitting_behind_validation",
        jobs=[],
        release_sha256=sha(out / "release.json"),
        handoff_source_sha256=sha(Path(__file__)),
        validation_dependencies=[cpu, cuda],
        superseded_controller=old_controller,
    )
    atomic(target, record)
    # The receipt prevents duplicate submissions even if the old controller races.
    scheduler(["/usr/bin/scancel", old_controller])
    record["controller_retired_without_scientific_execution"] = True
    atomic(target, record)
    for task in tasks:
        if gpu_nodes() >= release["scope"]["gpu_cap"]:
            raise RuntimeError("All-user twenty-GPU queued/running cap reached")
        args = [
            "/usr/bin/sbatch",
            "--parsable",
            "--account=IRI23021",
            "--nodes=1",
            "--ntasks=1",
            "--partition=gh",
            "--cpus-per-task=16",
            "--time=48:00:00",
            f"--dependency=afterok:{cpu}:{cuda}",
            str(root / "tools/empirical_checkpoint_gpu_v3.slurm"),
            "trajectory",
            str(task["index"]),
        ]
        stdout = scheduler(args)
        match = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$", stdout, re.M)
        if len(match) != 1:
            raise RuntimeError("Uncertain submission; inspect before retry")
        record["jobs"].append(
            dict(
                index=task["index"],
                stream=task["stream"],
                seed=task["seed"],
                job_id=match[0],
            )
        )
        atomic(target, record)
    dependencies = ":".join(j["job_id"] for j in record["jobs"])
    stdout = scheduler(
        [
            "/usr/bin/sbatch",
            "--parsable",
            "--account=IRI23021",
            "--nodes=1",
            "--ntasks=1",
            "--partition=gg",
            "--cpus-per-task=72",
            "--time=1:00:00",
            "--dependency=afterany:" + dependencies,
            str(root / "tools/empirical_checkpoint_cpu_v3.slurm"),
            "confirm",
        ]
    )
    match = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$", stdout, re.M)
    if len(match) != 1:
        raise RuntimeError("Uncertain confirmation submission; inspect receipt")
    record.update(
        status="queued_behind_both_validations_with_review_stop",
        confirmation_job=match[0],
    )
    atomic(target, record)
    print(json.dumps(record), flush=True)
    # Fail closed if a validation fails: cancel only this handoff's blocked GPU
    # entries so their afterany confirmation can report missing coverage.
    deadline = datetime.fromisoformat(release["scope"]["deadline_utc"])
    while datetime.now(timezone.utc) < deadline:
        raw = scheduler(
            [
                "/usr/bin/sacct",
                "-j",
                f"{cpu},{cuda}",
                "--format=JobIDRaw,State",
                "-n",
                "-P",
            ]
        )
        statuses = {
            line.split("|")[0]: line.split("|")[1].split()[0]
            for line in raw.splitlines()
            if "|" in line
        }
        if all(statuses.get(j) == "COMPLETED" for j in (cpu, cuda)):
            atomic(
                out / "validation_barrier_status.json",
                dict(status="both_validations_completed", validation_jobs=[cpu, cuda]),
            )
            return
        failed = {
            "FAILED",
            "TIMEOUT",
            "CANCELLED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "BOOT_FAIL",
        }
        if any(statuses.get(j) in failed for j in (cpu, cuda)):
            for job in record["jobs"]:
                if state(job["job_id"]) == "PENDING":
                    scheduler(["/usr/bin/scancel", job["job_id"]])
            atomic(
                out / "validation_barrier_status.json",
                dict(
                    status="validation_failed_no_collection", validation_states=statuses
                ),
            )
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
