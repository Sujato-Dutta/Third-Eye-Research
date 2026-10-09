"""E2 scheduler-only login relay; bounded jobs, release, deadline and GPU cap."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import empirical_relay_v1 as broker

OPTION = re.compile(
    r"^(?:--parsable|--partition=(?:gh|gg)|--account=IRI23021|--nodes=1|--ntasks=1|--cpus-per-task=(?:16|72)|--dependency=afterany:[0-9]+(?::[0-9]+)*|--time=(?:1|24|48):00:00)$"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(request, root):
    args = request["arguments"]
    if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
        raise ValueError("String scheduler arguments required")
    scripts = [i for i, a in enumerate(args) if a.startswith(str(root / "tools"))]
    if len(scripts) != 1:
        raise ValueError("Exactly one E2 script required")
    i = scripts[0]
    launcher = Path(args[i]).relative_to(root).as_posix()
    options, modes = args[:i], args[i + 1 :]
    if (
        any(not OPTION.fullmatch(a) for a in options)
        or len({a.split("=", 1)[0] for a in options}) != len(options)
        or any(
            options.count(a) != 1
            for a in ("--parsable", "--account=IRI23021", "--nodes=1", "--ntasks=1")
        )
    ):
        raise ValueError("Bounded scheduler options required")
    gpu = launcher == "tools/empirical_checkpoint_gpu_v3.slurm"
    if gpu:
        if (
            len(modes) != 2
            or modes[0] not in {"trajectory", "noise"}
            or not re.fullmatch(r"[0-7]", modes[1])
            or "--partition=gh" not in options
            or "--cpus-per-task=16" not in options
            or "--time=" + ("48:00:00" if modes[0] == "trajectory" else "24:00:00")
            not in options
        ):
            raise ValueError("Invalid E2 GPU task")
    elif (
        launcher != "tools/empirical_checkpoint_cpu_v3.slurm"
        or modes not in [["launch"], ["confirm"], ["review"]]
        or "--partition=gg" not in options
        or "--cpus-per-task=72" not in options
        or "--time=1:00:00" not in options
    ):
        raise ValueError("Invalid E2 CPU task")
    out = root / "runs/a2/empirical_v1/checkpoint_fallback_v3"
    release = json.loads((out / "release.json").read_text())
    if any(sha(root / name) != value for name, value in release["files"].items()):
        raise ValueError("Frozen E2 release differs")
    if any(
        sha(Path(path)) != value for path, value in release["prerequisites"].items()
    ):
        raise ValueError("Frozen E2 prerequisite differs")
    for kind in ("cpu", "cuda"):
        receipt = json.loads((out / f"{kind}_passed.json").read_text())
        if receipt["status"] != "passed" or receipt["release_sha256"] != sha(
            out / "release.json"
        ):
            raise ValueError("Both current CPU and CUDA validations required")
    if gpu:
        if datetime.now(timezone.utc) >= datetime.fromisoformat(
            release["scope"]["deadline_utc"]
        ):
            raise ValueError("Scientific experiment deadline passed")
        if (
            modes[0] == "noise"
            and not (out / "noise_inputs" / f"task_{modes[1]}.json").is_file()
        ):
            raise ValueError("Noise requires the actual predeclared retained state")
        queued = subprocess.check_output(
            ["/usr/bin/squeue", "-h", "-u", "sujato_ts", "-o", "%P|%T|%D"], text=True
        )
        nodes = sum(
            int(line.split("|")[2])
            for line in queued.splitlines()
            if set(line.split("|")[0].split(",")) & {"gh", "gh-dev"}
        )
        if nodes >= release["scope"]["gpu_cap"]:
            raise ValueError(
                "Twenty-GPU queued/running cap reached, including recovery"
            )
    if Path(request["cwd"]).resolve() != root.resolve():
        raise ValueError("Wrong project directory")
    if (
        not isinstance(request["environment"], dict)
        or set(request["environment"]) - broker.ENVIRONMENT_KEYS
        or any(not isinstance(v, str) for v in request["environment"].values())
    ):
        raise ValueError("Unexpected submission environment")


submit = broker.submit


def main():
    broker.validate = validate
    return broker.main()


if __name__ == "__main__":
    sys.exit(main())
