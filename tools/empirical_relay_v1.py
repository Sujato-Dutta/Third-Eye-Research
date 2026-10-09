"""Relay scheduler submissions only; scientific work stays in compute jobs.

The login-node service accepts a bounded set of sbatch requests from the
compute-node controller over the account's shared scratch filesystem.
It never imports the research package or runs model/evaluation code.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid


ENVIRONMENT_KEYS = {
    "HF_HOME",
    "THIRD_EYE_CONFIG",
    "THIRD_EYE_MANIFEST",
    "THIRD_EYE_GATE2",
    "THIRD_EYE_PLAN",
    "THIRD_EYE_STAGE",
    "THIRD_EYE_MODEL_SOURCES",
    "THIRD_EYE_SANDBOX_CONFIG",
    "THIRD_EYE_VERIFIER",
    "THIRD_EYE_VERIFIER_WORKERS",
    "THIRD_EYE_SYMBOLIC_WORKERS",
    "THIRD_EYE_VENV",
    "THIRD_EYE_HF_HOME",
    "THIRD_EYE_OMP_THREADS",
    "THIRD_EYE_SUBMISSION_QUEUE",
    "THIRD_EYE_EMPIRICAL_QUEUE",
    "THIRD_EYE_ORIGINAL_ROOT",
    "THIRD_EYE_A1_ROOT",
}
SCRIPTS = {"tools/empirical_noise_v1.slurm", "tools/empirical_noise_control_v1.slurm"}
OPTION = re.compile(
    r"^(?:--parsable|--partition=(?:gh|gg)|--account=IRI23021|"
    r"--nodes=1|--ntasks=1|--cpus-per-task=(?:16|72)|"
    r"--dependency=afterany:[0-9]+(?::[0-9]+)*|--time=(?:1|24):00:00)$"
)


def atomic_json(path, record):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def validate(request, root):
    arguments = request["arguments"]
    if not isinstance(arguments, list) or any(
        not isinstance(a, str) for a in arguments
    ):
        raise ValueError("Scheduler arguments must be strings")
    scripts = [i for i, a in enumerate(arguments) if a.startswith(str(root / "tools"))]
    if len(scripts) != 1:
        raise ValueError("Exactly one empirical launcher required")
    i = scripts[0]
    launcher = Path(arguments[i]).relative_to(root).as_posix()
    options, modes = arguments[:i], arguments[i + 1 :]
    if (
        launcher not in SCRIPTS
        or any(not OPTION.fullmatch(a) for a in options)
        or len(options) != len(set(options))
        or len({a.split("=", 1)[0] for a in options}) != len(options)
        or options.count("--account=IRI23021") != 1
        or options.count("--parsable") != 1
        or options.count("--nodes=1") != 1
        or options.count("--ntasks=1") != 1
    ):
        raise ValueError("Only bounded empirical submissions accepted")
    if launcher.endswith("noise_v1.slurm"):
        if (
            len(modes) != 1
            or not re.fullmatch(r"(?:[0-9]|1[01])", modes[0])
            or "--partition=gh" not in options
            or "--cpus-per-task=16" not in options
            or "--time=24:00:00" not in options
        ):
            raise ValueError("Invalid empirical GPU request")
    elif (
        modes not in [["review"]]
        or "--partition=gg" not in options
        or "--cpus-per-task=72" not in options
        or not any(a in options for a in ["--time=1:00:00"])
    ):
        raise ValueError("Invalid empirical CPU request")
    release = json.loads((root / "runs/a2/empirical_v1/noise_release.json").read_text())
    import hashlib

    if not all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
        for name, sha in release["files"].items()
    ):
        raise ValueError("Empirical release hashes changed")
    cached_path = root / "runs/a2/empirical_v1/release.json"
    if (
        hashlib.sha256(cached_path.read_bytes()).hexdigest()
        != release["cached_release_sha256"]
    ):
        raise ValueError("Cached release changed")
    cached = json.loads(cached_path.read_text())
    if (
        hashlib.sha256(
            (root / "runs/a2/empirical_v1/scope.json").read_bytes()
        ).hexdigest()
        != cached["scope_sha256"]
    ):
        raise ValueError("Empirical scope changed")
    scope = json.loads((root / "runs/a2/empirical_v1/scope.json").read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(scope["deadline_utc"]):
        raise ValueError("Empirical deadline passed")
    if launcher.endswith("noise_v1.slurm"):
        passed = json.loads(
            (root / "runs/a2/empirical_v1/gpu_verification/passed.json").read_text()
        )
        if (
            passed["noise_release_sha256"]
            != hashlib.sha256(
                (root / "runs/a2/empirical_v1/noise_release.json").read_bytes()
            ).hexdigest()
        ):
            raise ValueError("GPU replay validation release differs")
        queued = subprocess.check_output(
            ["/usr/bin/squeue", "-h", "-u", "sujato_ts", "-o", "%P|%T|%D"], text=True
        )
        gpu_nodes = sum(
            int(line.split("|")[2])
            for line in queued.splitlines()
            if set(line.split("|")[0].split(",")) & {"gh", "gh-dev"}
        )
        if gpu_nodes >= scope["gpu_cap"]:
            raise ValueError(
                "Twenty-GPU queued/running ceiling reached; inspect before submitting"
            )
    if Path(request["cwd"]).resolve() != root.resolve():
        raise ValueError("Submission directory must be the deployed project")
    if (
        not isinstance(request["environment"], dict)
        or set(request["environment"]) - ENVIRONMENT_KEYS
    ):
        raise ValueError("Submission environment contains unexpected fields")
    if any(not isinstance(v, str) for v in request["environment"].values()):
        raise ValueError("Environment values must be strings")


def submit(arguments):
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Relay client requires a scheduled compute-node controller")
    queue = Path(os.environ["THIRD_EYE_EMPIRICAL_QUEUE"])
    name = uuid.uuid4().hex
    request = queue / (name + ".request.json")
    response = queue / (name + ".response.json")
    atomic_json(
        request,
        {
            "arguments": arguments,
            "cwd": str(Path.cwd()),
            "environment": {
                k: v for k, v in os.environ.items() if k in ENVIRONMENT_KEYS
            },
            "controller_job": os.environ["SLURM_JOB_ID"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    deadline = time.monotonic() + 180
    while not response.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(
                "Submission relay timed out; inspect request before retrying to avoid duplicates: "
                + name
            )
        time.sleep(1)
    record = json.loads(response.read_text())
    if record["exit_code"]:
        raise RuntimeError(
            "Login-node scheduler relay rejected: "
            + record.get("stdout", "")
            + record.get("stderr", "")
        )
    return record["stdout"]


def serve(root, queue, deadline):
    if "login" not in socket.gethostname().lower() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scheduler-only relay must run on a login node")
    import fcntl

    queue.mkdir(parents=True, exist_ok=True)
    lock = (queue / "broker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = {
        k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))
    }
    heartbeat = 0
    while datetime.now(timezone.utc) < deadline:
        if time.monotonic() > heartbeat:
            atomic_json(
                queue / "broker_status.json",
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "role": "sbatch submissions only",
                },
            )
            heartbeat = time.monotonic() + 60
        for pending in sorted(queue.glob("*.request.json")):
            response = pending.with_name(
                pending.name.replace(".request.json", ".response.json")
            )
            if response.exists():
                continue
            # A claimed request is never automatically retried after a crash.
            claimed = pending.with_name(
                pending.name.replace(".request.json", ".claimed.json")
            )
            pending.replace(claimed)
            try:
                request = json.loads(claimed.read_text())
                validate(request, root)
                completed = subprocess.run(
                    ["/usr/bin/sbatch", *request["arguments"]],
                    cwd=root,
                    env={**base, **request["environment"]},
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                record = {
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            except Exception as error:
                record = {"exit_code": 1, "stdout": "", "stderr": str(error)}
            record["submitted_from"] = socket.gethostname()
            record["processed_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(response, record)
        time.sleep(5)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        print(submit(sys.argv[2:]), end="")
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("Deadline must include timezone")
    serve(args.root.resolve(), args.queue.resolve(), deadline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
