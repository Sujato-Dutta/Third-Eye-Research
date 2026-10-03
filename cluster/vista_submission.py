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
}
SCRIPTS = {"experiments/jobs/vista_pilot.slurm", "experiments/jobs/vista_study.slurm"}
OPTION = re.compile(
    r"^(?:--parsable|--test-only|--partition=(?:gh|gh-dev|gg)|--account=IRI23021|"
    r"--array=[0-9,%\-]+|--dependency=afterok:[0-9:]+|--time=[0-9:]+)$"
)


def atomic_json(path, record):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def validate(request, root):
    arguments = request["arguments"]
    if (
        not isinstance(arguments, list)
        or not arguments
        or arguments[-1] not in SCRIPTS
        or any(
            not isinstance(a, str) or not OPTION.fullmatch(a) for a in arguments[:-1]
        )
        or arguments.count("--account=IRI23021") != 1
        or arguments.count("--parsable") != 1
    ):
        raise ValueError("Only bounded Vista project sbatch submissions are accepted")
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
    queue = Path(os.environ["THIRD_EYE_SUBMISSION_QUEUE"])
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
    sys.stdout.write(record.get("stdout", ""))
    sys.stderr.write(record.get("stderr", ""))
    return record["exit_code"]


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
        return submit(sys.argv[2:])
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
