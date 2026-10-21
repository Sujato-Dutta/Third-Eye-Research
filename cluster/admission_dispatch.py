"""Login-only Slurm dispatch of an already compute-validated admission plan.

No model, dataset, numerical or statistical modules are imported here. Vista
requires scheduler submission through login resources. Computational work stays
inside the source-bound Slurm jobs referenced by the validated plan.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def login_only():
    if "login" not in socket.gethostname().lower() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Vista scheduler dispatch must run on a login resource, outside a compute allocation"
        )


def check_handoff(output, expected_source, expected_plan):
    """Accept a prepared handoff or the explicitly reviewed legacy host denial."""
    output = Path(output)
    ready = output / "dispatch_ready.json"
    if ready.exists():
        value = read(ready)
        if (
            value["source_tree_sha256"] != expected_source
            or value["plan_sha256"] != expected_plan
            or value["parents"] != 16
            or value["scheduler_dispatch_host"] != "login_resource_only"
        ):
            raise ValueError("Scheduler handoff differs from the validated plan")
        return "prepared_handoff"
    failure = read(output / "llama-3.2-3b-instruct_code_response.json")
    if (
        failure["returncode"] != 1
        or "Job submission is not allowed from this host" not in failure["stdout"]
    ):
        raise ValueError("Recovery requires an explicit pre-submission host rejection")
    return "reviewed_legacy_host_denial"


def submit_job(command, env, record):
    """Persist the attempt before calling Slurm; never retry an ambiguous result."""
    from third_eye.io import write_json

    path = Path(record)
    if path.exists():
        raise RuntimeError("Dispatch attempt already exists; audit it before any retry")
    write_json(path, {"command": command, "status": "attempting"})
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    write_json(
        path,
        {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    ids = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$", result.stdout, re.M)
    if result.returncode or len(ids) != 1:
        raise RuntimeError("Submission failed/ambiguous; preserve reservation and stop")
    return ids[0]


def dispatch(request_path):
    login_only()
    request = read(request_path)
    output, source = Path(request["output"]), Path(request["source"])
    sys.path.insert(0, str(source / "src"))
    from third_eye.cluster_budget import reconcile
    from third_eye.io import write_json

    plan, release = read(output / "plan.json"), read(source / "release.json")
    expected = request["source_tree_sha256"]
    if (
        plan["source_tree_sha256"] != expected
        or release["source_tree_sha256"] != expected
    ):
        raise ValueError("Validated plan/source release differs")
    for path, expected_hash in release["files"].items():
        if sha256(source / path) != expected_hash:
            raise ValueError("Immutable compute source changed")
    for key in ("cpu_validation", "cuda_validation"):
        value = read(plan[key])
        if value.get("passed") is not True or value["source_tree_sha256"] != expected:
            raise ValueError("Matching successful compute validations required")
    preflight = read(Path(plan["cuda_validation"]).parent / "real_backbones.json")
    if preflight.get("passed") is not True or len(preflight["models"]) != 4:
        raise ValueError("All four real-backbone stream preflights must pass")
    if plan["purpose"] != "admission_development_pilots" or len(plan["tasks"]) != 16:
        raise ValueError(
            "Only the frozen sixteen-parent development pilot is authorized"
        )
    for task in plan["tasks"]:
        if (
            sha256(task["config"]) != task["config_sha256"]
            or sha256(task["manifest"]) != task["manifest_sha256"]
        ):
            raise ValueError("Configuration or manifest changed")
        if Path(task["output"]).exists():
            raise ValueError(
                "Existing pilot output: stop rather than duplicate a parent"
            )
    receipt = read(output / "submissions.json")
    if receipt.get("pilot_arrays") or receipt.get("review"):
        raise ValueError("Pilot submission already recorded; do not duplicate jobs")
    handoff = check_handoff(output, expected, sha256(output / "plan.json"))
    streams = sorted(plan["runtime_limits_by_stream"])
    for stream in streams[1:]:
        if (output / (stream.replace("/", "_") + "_response.json")).exists():
            raise ValueError(
                "Unexpected later controller attempt; audit before dispatch"
            )

    ledger_path, prefix = Path(request["ledger"]), request["ledger_prefix"]
    ledger = read(ledger_path)
    if ledger["hard_cap_sus"] != 1200:
        raise ValueError("Persistent project cap changed")
    ids = {value["job"] for value in receipt["jobs"].values()}
    result = subprocess.run(
        [
            "sacct",
            "--array",
            "-j",
            ",".join(sorted(ids)),
            "-X",
            "-P",
            "--format=JobID,State,ElapsedRaw,AllocNodes",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    accounting = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0] in ids:
            accounting[fields[0]] = {
                "id": fields[0],
                "state": fields[1],
                "elapsed_seconds": int(fields[2]),
                "nodes": int(fields[3]),
            }
    if accounting.keys() != ids:
        raise ValueError("Complete predecessor accounting required")
    for name, job in receipt["jobs"].items():
        reconcile(ledger, prefix + name, [accounting[job["job"]]])
    for stream in streams:
        job = ledger["jobs"][prefix + stream.replace("/", "_")]
        if (
            job["status"] != "reserved"
            or job.get("job_id")
            or job["tasks"] != 4
            or job["seconds"] != plan["runtime_limits_by_stream"][stream]
        ):
            raise ValueError(
                "Reuse only the existing exact, unsubmitted array reservation"
            )
    review_reservation = ledger["jobs"][prefix + "review"]
    if review_reservation["status"] != "reserved" or review_reservation.get("job_id"):
        raise ValueError("Review reservation already submitted")
    stage = sum(
        j.get("actual_sus", j["reserved_sus"])
        for k, j in ledger["jobs"].items()
        if k.startswith("admission_b_")
    )
    total = sum(j.get("actual_sus", j["reserved_sus"]) for j in ledger["jobs"].values())
    if stage > 192 + 1e-9 or total > 1200 + 1e-9:
        raise ValueError("Stage/project cap would be exceeded")
    environment = read(output / "environment.json")
    env = dict(
        os.environ,
        THIRD_EYE_SOURCE_ROOT=str(source),
        THIRD_EYE_VENV=str(Path(request["root"]) / ".venv"),
        THIRD_EYE_PILOT_PLAN=str(output / "plan.json"),
        THIRD_EYE_ADMISSION_REQUEST=str(request_path),
        HF_HOME=environment["HF_HOME"],
    )
    for key, value in environment["bound_configs"].items():
        if sha256(value["path"]) != value["sha256"]:
            raise ValueError("Bound deployment environment changed")
        env[key] = value["path"]
    write_json(ledger_path, ledger)
    audit_path = output / "login_dispatch.json"
    if audit_path.exists():
        raise ValueError("Login dispatch already attempted; no automatic retry")
    audit = {
        "host": socket.gethostname(),
        "dispatcher_sha256": sha256(__file__),
        "source_tree_sha256": expected,
        "plan_sha256": sha256(output / "plan.json"),
        "predecessor_accounting": accounting,
        "stage_b_charged_plus_reserved_sus": stage,
        "project_charged_plus_reserved_sus": total,
        "jobs": {},
        "automatic_expansion": False,
        "handoff_kind": handoff,
    }
    write_json(audit_path, audit)
    dependencies = []
    for index, stream in enumerate(streams):
        name = stream.replace("/", "_")
        command = [
            "sbatch",
            "--parsable",
            "--chdir",
            str(source),
            "--partition",
            "gh",
            "--time",
            str(plan["runtime_limits_by_stream"][stream] // 60),
            "--array=0-3%4",
            "--export=ALL",
            "--output",
            str(output / (name + "_%A_%a.out")),
            str(source / "experiments/jobs/forecast_pilots.slurm"),
        ]
        job = submit_job(
            command,
            dict(env, THIRD_EYE_INDEX_BASE=str(index * 4)),
            output / (name + "_login_attempt.json"),
        )
        ledger["jobs"][prefix + name].update(job_id=job, status="submitted")
        write_json(ledger_path, ledger)
        record = {
            "job": job,
            "index_base": index * 4,
            "seconds": plan["runtime_limits_by_stream"][stream],
        }
        receipt.setdefault("pilot_arrays", {})[stream] = record
        audit["jobs"][stream] = record
        write_json(output / "submissions.json", receipt)
        write_json(audit_path, audit)
        dependencies.append(job)
    command = [
        "sbatch",
        "--parsable",
        "--chdir",
        str(source),
        "--partition",
        "gg",
        "--time",
        "60",
        "--dependency=afterany:" + ":".join(dependencies),
        "--export=ALL",
        "--output",
        str(output / "review_%A_%a.out"),
        str(source / "experiments/jobs/admission_pilot_review.slurm"),
    ]
    job = submit_job(command, env, output / "review_login_attempt.json")
    ledger["jobs"][prefix + "review"].update(job_id=job, status="submitted")
    write_json(ledger_path, ledger)
    receipt["review"] = job
    audit["review"] = job
    write_json(output / "submissions.json", receipt)
    write_json(audit_path, audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    dispatch(parser.parse_args().request)
