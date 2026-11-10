"""Allocated setup, resource admission and execution for the horizon extension."""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile

PRIMARY = Path("/dgxa_home/se23uari167/third_eye_dgx_20261026")
ROOT = Path("/dgxa_home/se23ucse205/third_eye_horizon_20261109")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(path).with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temp.replace(path)


def source():
    return ROOT/("release_"+read(ROOT/"transfer.json")["source_tree_sha256"][:12])


def imports():
    sys.path[:0] = [str(source()/"src"), str(source()/"experiments")]
    importlib.invalidate_caches()
    os.environ["PYTHONPATH"] = str(source()/"src")+":"+str(source()/"experiments")
    import long_horizon_trajectories as runner
    return runner


def validate():
    metadata = read(ROOT/"transfer.json")
    if sha(ROOT/"source.tar.gz") != metadata["archive_sha256"] or sha(ROOT/"preregistration.md") != metadata["preregistration_sha256"]:
        raise ValueError("Transferred source or preregistration differs")
    if sha(__file__) != metadata["driver_sha256"]:
        raise ValueError("Pinned operational driver differs")
    dest = source()
    dest.mkdir(exist_ok=True)
    with tarfile.open(ROOT/"source.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not (dest/member.name).resolve().is_relative_to(dest.resolve()):
                raise ValueError("Unsafe source archive")
        for member in archive.getmembers():
            path = dest/member.name
            data = archive.extractfile(member).read()
            if path.exists():
                if path.is_symlink() or sha(path) != hashlib.sha256(data).hexdigest():
                    raise ValueError("Previously extracted source differs")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    imports()
    from third_eye.provenance import source_inventory
    if source_inventory(dest)["source_tree_sha256"] != metadata["source_tree_sha256"]:
        raise ValueError("Scientific source differs")
    validation = ROOT/"validation"
    validation.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", str(validation/"fixtures")], cwd=dest, check=True)
    subprocess.run([sys.executable, "-m", "ruff", "check", "src", "experiments", "tests"], cwd=dest, check=True)
    write(validation/"source_completed.json", {"passed": True, "source_tree_sha256": metadata["source_tree_sha256"],
        "archive_sha256": metadata["archive_sha256"], "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname()})


def bind():
    runner = imports()
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.final import load_final_manifest
    from third_eye.experiments.long_horizon import configured
    from third_eye.config import Config
    from next_round_regimes import verify_models
    metadata = read(ROOT/"transfer.json")
    models = read(ROOT/"models_completed.json")
    validated = read(ROOT/"validation/source_completed.json")
    if not models["passed"] or not validated["passed"] or sha(ROOT/"model_sources.json") != models["model_sources_sha256"]:
        raise ValueError("Complete validated source and byte-verified model preparation required")
    primary_path = PRIMARY/"runs/positive_regime_20261109/request.json"
    primary = read(primary_path)
    if sha(primary_path) != metadata["primary_request_sha256"]:
        raise ValueError("Original confirmation request differs")
    request = {"declaration": runner.DECLARATION, "source": str(source()), "output": str(ROOT),
        "source_tree_sha256": metadata["source_tree_sha256"], "template": primary["config"],
        "manifest": primary["manifest"], "final_manifest": primary["final_manifest"],
        "primary_request": str(primary_path), "model_sources": str(ROOT/"model_sources.json"),
        "validation": str(ROOT/"validation/completed.json"), "deadline_utc": metadata["deadline_utc"],
        "preregistration_sha256": metadata["preregistration_sha256"]}
    for field in ("template", "manifest", "final_manifest", "primary_request", "model_sources"):
        request[field+"_sha256"] = sha(request[field])
    write(ROOT/"request.json", request)
    runner.checked(ROOT/"request.json", validation=False)
    load_manifest(request["manifest"])
    load_final_manifest(request["final_manifest"], request["manifest"])
    tasks = []
    for model, spec in runner.MODELS.items():
        cfg = configured(Config.load(request["template"]), model, spec["seeds"][0])
        path = ROOT/"configs"/(model+".json")
        write(path, cfg.to_dict())
        tasks.append({"config": str(path)})
    os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
    checks = verify_models({"reference_tasks": tasks})
    write(ROOT/"validation/completed.json", {**validated, "passed": True,
        "request_sha256": sha(ROOT/"request.json"), "model_checks": checks,
        "model_preparation_sha256": sha(ROOT/"models_completed.json"), "job_id": os.environ["SLURM_JOB_ID"]})


def guard(model=None, compatibility=False):
    request = read(ROOT/"request.json")
    deadline = datetime.datetime.fromisoformat(request["deadline_utc"].replace("Z", "+00:00"))
    remaining = (deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()/3600
    hours = .5 if compatibility else imports().MODELS[model]["gpu_hours"] if model else 0
    if remaining < hours:
        raise RuntimeError("Job's protective runtime no longer fits the 20-day execution window")
    subprocess.run([sys.executable, str(ROOT/"storage.py"), "check", "--output",
        str(Path.home()/"third_eye_account_audit_20261109"), "--reserve-bytes", "2000000000",
        "--limit-bytes", "50000000000", "--warning-bytes", "45000000000",
        "--admission-limit-bytes", "48000000000"], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "bind", "compatibility", "run", "archive", "review", "final"))
    parser.add_argument("--model")
    parser.add_argument("--index", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower() or
            Path.home().resolve() != ROOT.parent or os.getuid() != 1339):
        raise RuntimeError("Named additional account and Slurm compute node required")
    if args.mode == "validate":
        validate()
        return
    if args.mode == "bind":
        bind()
        return
    runner = imports()
    request = runner.checked(ROOT/"request.json")
    os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
    if args.mode == "compatibility":
        guard(args.model, compatibility=True)
        runner.compatibility(request, args.model)
    elif args.mode == "run":
        guard(args.model)
        runner.run(request, args.model, args.index)
    elif args.mode == "archive":
        from third_eye.experiments.scale import compact_checkpoints
        runner.inspect(request, args.model, args.index)
        compact_checkpoints(ROOT/"parents"/args.model/str(runner.MODELS[args.model]["seeds"][args.index]))
    elif args.mode == "review":
        runner.review(request, args.model)
    else:
        reviews = {m: read(ROOT/(m+"_review.json")) for m in runner.MODELS}
        if not all(r["passed"] and r["request_sha256"] == sha(ROOT/"request.json") for r in reviews.values()):
            raise ValueError("Complete four-model review required")
        write(ROOT/"completed.json", {"passed": True, "models": reviews, "declaration": runner.DECLARATION,
            "request_sha256": sha(ROOT/"request.json"), "forecaster_fits": 0, "automatic_expansion": False,
            "interpretation": "paired longer-horizon sensitivity; four seeds/backbone, not a precision guarantee"})


if __name__ == "__main__":
    main()
