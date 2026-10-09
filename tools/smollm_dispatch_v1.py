"""Scheduler-only S1 dispatcher reserves all future core reliability slots."""

from datetime import datetime, timezone, timedelta
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


def live_queue(raw):
    return {
        line.split("|")[0]: dict(
            partition=line.split("|")[1],
            state=line.split("|")[2],
            nodes=int(line.split("|")[3]),
        )
        for line in raw.splitlines()
        if line.strip()
    }


def reserved_core_slots(core, queued):
    preferred = core / "path_repair_v1/submissions.json"
    path = preferred if preferred.exists() else core / "trajectory_submissions.json"
    jobs = json.loads(path.read_text())["jobs"]
    reserve = 0
    for job in jobs:
        receipt = core / "noise_submissions" / f"task_{job['index']}.json"
        submitted = (
            receipt.exists()
            and json.loads(receipt.read_text()).get("status") == "queued"
        )
        may_produce = (
            job["job_id"] in queued
            or (core / "noise_inputs" / f"task_{job['index']}.json").exists()
        )
        reserve += int(not submitted and may_produce)
    return reserve


def capacity(core, queued, cap=20):
    used = sum(
        j["nodes"]
        for j in queued.values()
        if set(j["partition"].split(",")) & {"gh", "gh-dev"}
    )
    reserve = reserved_core_slots(core, queued)
    return dict(
        occupied_gpu_nodes=used,
        reserved_core_controls=reserve,
        free_after_reservation=max(0, cap - used - reserve),
    )


def scheduler(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def submit(root, mode, index=None, dependency=None):
    gpu = mode in {"verify", "run"}
    args = [
        "/usr/bin/sbatch",
        "--parsable",
        "--account=IRI23021",
        "--nodes=1",
        "--ntasks=1",
        "--partition=" + ("gh-dev" if mode == "verify" else "gh" if gpu else "gg"),
        "--cpus-per-task=" + ("16" if gpu else "72"),
        "--time="
        + ({"verify": "1:00:00", "run": "12:00:00", "review": "1:00:00"}[mode]),
    ]
    if dependency:
        args.append(
            "--dependency="
            + ("afterany:" if mode == "review" else "afterok:")
            + ":".join(dependency)
        )
    elif mode == "verify":
        args.append("--dependency=afterok:1057730")
    args += [str(root / f"tools/smollm_{'gpu' if gpu else 'cpu'}_v1.slurm"), mode]
    if index is not None:
        args.append(str(index))
    stdout = scheduler(args)
    matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$", stdout, re.M)
    if len(matches) != 1:
        raise RuntimeError("Uncertain scheduler submission; inspect before retry")
    return matches[0]


def passed_receipt(path, release_sha):
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    return (
        receipt.get("status") == "passed"
        and receipt.get("release_sha256") == release_sha
    )


def main():
    if "login" not in socket.gethostname().lower() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scheduler-only S1 service requires login host")
    root = Path.cwd()
    core = root / "runs/a2/empirical_v1/checkpoint_fallback_v3"
    out = root / "runs/a2/empirical_v1/smollm_supplement_v1"
    release_path = out / "release.json"
    release = json.loads(release_path.read_text())
    release_sha = sha(release_path)
    deadline = datetime.fromisoformat(release["scope"]["deadline_utc"])
    last_submit = deadline - timedelta(hours=14)
    core_sha = sha(core / "release.json")
    if core_sha != release["core_release_sha256"]:
        raise RuntimeError("S1 core release differs")
    if any(sha(root / p) != h for p, h in release["files"].items()):
        raise RuntimeError("S1 sources differ")
    target = out / "submissions.json"
    if target.exists():
        raise RuntimeError("S1 dispatcher previously attempted work; inspect receipts")
    record = dict(
        status="waiting_for_setup_and_reserved_capacity",
        jobs=[],
        release_sha256=release_sha,
        deadline_utc=deadline.isoformat(),
    )
    atomic(target, record)
    while datetime.now(timezone.utc) < last_submit:
        queued = live_queue(
            scheduler(["/usr/bin/squeue", "-h", "-u", "sujato_ts", "-o", "%i|%P|%T|%D"])
        )
        cap = capacity(core, queued)
        record["capacity"] = cap
        record["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic(target, record)
        if (out / "setup/failure.json").exists() or (
            out / "cuda_verification/failure.json"
        ).exists():
            record["status"] = "compatibility_failed_no_collection"
            atomic(target, record)
            return
        cpu = passed_receipt(out / "cpu_passed.json", release_sha)
        core_ok = all(
            passed_receipt(core / f"{kind}_passed.json", core_sha)
            for kind in ("cpu", "cuda")
        )
        repair = core / "path_repair_v1"
        repair_ok = (repair / "release.json").exists() and passed_receipt(
            repair / "cpu_passed.json", sha(repair / "release.json")
        )
        if (
            cpu
            and core_ok
            and repair_ok
            and "cuda_verification_job" not in record
            and cap["free_after_reservation"]
        ):
            record["status"] = "submitting_cuda_verification"
            atomic(target, record)
            record["cuda_verification_job"] = submit(root, "verify")
            record["status"] = "waiting_for_cuda_verification"
            atomic(target, record)
        cuda = passed_receipt(out / "cuda_passed.json", release_sha)
        if cpu and cuda and core_ok and repair_ok and len(record["jobs"]) < 4:
            # One submission per poll; recompute reservation before the next job.
            if cap["free_after_reservation"]:
                index = len(record["jobs"])
                record["status"] = "submitting_single_state"
                record["attempting_index"] = index
                atomic(target, record)
                job = submit(
                    root, "run", index, dependency=[record["cuda_verification_job"]]
                )
                record["jobs"].append(dict(index=index, job_id=job))
                record.pop("attempting_index")
                record["status"] = "collecting_as_core_capacity_frees"
                atomic(target, record)
        if len(record["jobs"]) == 4:
            record["review_job"] = submit(
                root, "review", dependency=[j["job_id"] for j in record["jobs"]]
            )
            record["status"] = "queued_four_states_and_review_stop"
            atomic(target, record)
            return
        # Detect scheduler-level preflight failure even without a Python receipt.
        if "cuda_verification_job" in record and not cuda:
            raw = scheduler(
                [
                    "/usr/bin/sacct",
                    "-j",
                    record["cuda_verification_job"],
                    "--format=JobIDRaw,State",
                    "-n",
                    "-P",
                ]
            )
            states = {
                line.split("|")[0]: line.split("|")[1].split()[0]
                for line in raw.splitlines()
                if "|" in line
            }
            if states.get(record["cuda_verification_job"]) in {
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
            }:
                record["status"] = "cuda_job_failed_no_collection"
                atomic(target, record)
                return
        if not cpu and (out / "setup_submission.json").exists():
            job = json.loads((out / "setup_submission.json").read_text())["job_id"]
            raw = scheduler(
                ["/usr/bin/sacct", "-j", job, "--format=JobIDRaw,State", "-n", "-P"]
            )
            states = {
                line.split("|")[0]: line.split("|")[1].split()[0]
                for line in raw.splitlines()
                if "|" in line
            }
            if states.get(job) in {
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
            }:
                record["status"] = "setup_job_failed_no_collection"
                atomic(target, record)
                return
        time.sleep(30)
    record["status"] = "submission_cutoff_partial_coverage"
    record["review_job"] = submit(
        root, "review", dependency=[j["job_id"] for j in record["jobs"]] or None
    )
    atomic(target, record)


if __name__ == "__main__":
    main()
