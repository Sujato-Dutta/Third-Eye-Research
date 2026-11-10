"""Read-only account storage accounting and admission guard, compute nodes only."""
import argparse
import datetime
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess
import time

HARD_LIMIT = 50_000_000_000
WARNING_LIMIT = 45_000_000_000
ADMISSION_LIMIT = 48_000_000_000


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def measure(root, apparent=False, depth=None):
    command = ["du", "-x", "-B1"]
    command += ["--apparent-size"] if apparent else []
    command += ["--max-depth="+str(depth)] if depth is not None else ["-s"]
    result = subprocess.run(command+[str(root)], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("Incomplete account storage scan; refuse to admit new writes")
    rows = []
    for line in result.stdout.splitlines():
        size, path = line.split("\t", 1)
        rows.append({"bytes": int(size), "path": path})
    return rows


def snapshot(output, reserve_bytes=0, breakdown=False):
    user = pwd.getpwuid(os.getuid()).pw_name
    home = Path.home()
    roots = []
    for candidate in (home, Path("/dgxa_home")/user, Path("/dgxb_home")/user):
        if not candidate.exists() or any(os.path.samefile(candidate, r) for r in roots):
            continue
        if candidate.stat().st_uid != os.getuid():
            raise RuntimeError("Account storage root ownership differs")
        roots.append(candidate)
    records = []
    for root in roots:
        allocated = measure(root)[0]["bytes"]
        apparent = measure(root, apparent=True)[0]["bytes"]
        record = {"path": str(root), "allocated_bytes": allocated, "apparent_bytes": apparent,
                  "conservative_bytes": max(allocated, apparent)}
        if breakdown:
            record["top_level_allocated"] = sorted(measure(root, depth=1),
                                                    key=lambda row: row["bytes"], reverse=True)
        records.append(record)
    total = sum(row["conservative_bytes"] for row in records)
    quota = {"available": shutil.which("quota") is not None}
    if quota["available"]:
        result = subprocess.run(["quota", "-s"], capture_output=True, text=True, timeout=30)
        quota.update({"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode})
    filesystem = subprocess.run(["df", "-B1", "-T", str(home)],
                                capture_output=True, text=True, check=True).stdout
    result = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user": user, "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"), "roots": records,
        "total_conservative_bytes": total, "policy_limit_bytes": HARD_LIMIT,
        "policy_headroom_bytes": HARD_LIMIT-total, "warning_bytes": WARNING_LIMIT,
        "admission_limit_bytes": ADMISSION_LIMIT, "reserve_bytes": reserve_bytes,
        "warning": total >= WARNING_LIMIT,
        "new_writes_allowed": total+reserve_bytes < ADMISSION_LIMIT,
        "under_hard_limit": total < HARD_LIMIT,
        "filesystem_quota": quota, "filesystem_capacity_not_account_quota": filesystem,
        "automatic_deletion": False, "automatic_job_cancellation": False}
    write(output/"storage_latest.json", result)
    with (output/"storage_history.jsonl").open("a") as stream:
        stream.write(json.dumps({k: result[k] for k in (
            "utc", "job_id", "host", "total_conservative_bytes", "policy_headroom_bytes",
            "warning", "new_writes_allowed", "under_hard_limit")})+"\n")
    print(json.dumps({"storage_gb": round(total/1e9, 3), "policy_limit_gb": HARD_LIMIT/1e9,
        "headroom_gb": round((HARD_LIMIT-total)/1e9, 3),
        "new_writes_allowed": result["new_writes_allowed"], "job_id": os.environ["SLURM_JOB_ID"]}), flush=True)
    return result


def main():
    global HARD_LIMIT, WARNING_LIMIT, ADMISSION_LIMIT
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "check", "watch"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--duration-hours", type=float, default=168)
    parser.add_argument("--interval-seconds", type=int, default=1200)
    parser.add_argument("--limit-bytes", type=int, default=HARD_LIMIT)
    parser.add_argument("--warning-bytes", type=int, default=WARNING_LIMIT)
    parser.add_argument("--admission-limit-bytes", type=int, default=ADMISSION_LIMIT)
    args = parser.parse_args()
    HARD_LIMIT, WARNING_LIMIT, ADMISSION_LIMIT = args.limit_bytes, args.warning_bytes, args.admission_limit_bytes
    if not 0 < WARNING_LIMIT < ADMISSION_LIMIT < HARD_LIMIT:
        raise ValueError("Storage thresholds must retain headroom under the hard limit")
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Storage scans require a Slurm compute allocation")
    if args.reserve_bytes < 0 or args.interval_seconds < 60 or args.duration_hours <= 0:
        raise ValueError("Invalid storage reservation or monitoring interval")
    output = Path(args.output)
    if output.resolve().parent != Path.home().resolve() or not output.name.startswith("third_eye_account_audit_"):
        raise ValueError("Audit outputs must stay in the dedicated account directory")
    output.mkdir(exist_ok=True)
    until = time.monotonic()+args.duration_hours*3600
    while True:
        try:
            result = snapshot(output, args.reserve_bytes, breakdown=args.mode == "snapshot")
        except Exception as error:
            write(output/"storage_failure.json", {"new_writes_allowed": False,
                "error_type": type(error).__name__, "message": str(error), "job_id": os.environ["SLURM_JOB_ID"]})
            raise
        if args.mode == "check" and not result["new_writes_allowed"]:
            raise SystemExit(2)
        if args.mode != "watch" or time.monotonic()+args.interval_seconds >= until:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
