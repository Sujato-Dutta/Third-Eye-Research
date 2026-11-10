"""Allocated validation, storage checks and execution of the frozen confirmation."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile

ROOT = Path("/dgxa_home/se23uari167/third_eye_dgx_20261026")
DEV = ROOT/"runs/next_round_dgx_20261108"
CAMPAIGN = ROOT/"runs/positive_regime_20261109"
LIMIT = 50_000_000_000


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


def usage(path=None):
    result = subprocess.run(["du", "-x", "-B1", "-s", str(path or Path.home())],
                            text=True, capture_output=True, check=True)
    return int(result.stdout.split()[0])


def retire_completed_llama_cache():
    """Recoverable base-weight cache only; verify all targets before any deletion."""
    active = subprocess.run(["squeue", "-h", "-u", str(os.getuid()), "-o", "%P|%T"],
                            text=True, capture_output=True, check=True).stdout
    if any("gpu" in row and "RUNNING" in row for row in active.splitlines()):
        raise ValueError("Cache retirement requires no running account GPU job")
    for seed in range(2026110600, 2026110604):
        parent = DEV/f"development/llama-3.2-3b-instruct/{seed}"
        if not read(parent/"completed.json")["passed"] or not read(parent/"lossless_retention.json")["passed"]:
            raise ValueError("Complete retained Llama development cohort required")
    registry_path = ROOT/"imported/registry.json"
    if sha(registry_path) != read(ROOT/"prepared.json")["registry_sha256"]:
        raise ValueError("Frozen pretrained registry differs")
    model = read(registry_path)["models"]["meta-llama/Llama-3.2-3B-Instruct"]
    cache_root = ROOT/"models/hf_cache"
    cache = cache_root/"hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"/model["revision"]
    if cache.is_symlink() or not cache.resolve().is_relative_to(cache_root.resolve()):
        raise ValueError("Unsafe cache root")
    records = []
    for name, expected in sorted(model["files"].items()):
        if not name.endswith(".safetensors"):
            continue
        path = cache/name
        target = path.resolve(strict=True)
        if (not target.is_relative_to(cache_root.resolve()) or
                target.stat().st_uid != os.getuid() or target.stat().st_size != expected["bytes"] or
                sha(target) != expected["sha256"]):
            raise ValueError("Cache target escaped allowlist or failed source verification")
        records.append({"path": str(path), "target": str(target), **expected})
    if not records:
        raise ValueError("No verified recoverable weights to retire")
    receipt = {"files": records, "before_home_bytes": usage(), "registry_sha256": sha(registry_path),
        "preserved": "all research artifacts, adapters, environments and model metadata",
        "job_id": os.environ["SLURM_JOB_ID"], "passed": False}
    write(CAMPAIGN/"recoverable_cache_rotation.json", receipt)
    for record in records:
        path, target = Path(record["path"]), Path(record["target"])
        # Both absolute targets have already been resolved and checked above.
        if path.is_symlink():
            path.unlink()
        if target.exists():
            target.unlink()
    receipt.update({"passed": True, "after_home_bytes": usage()})
    write(CAMPAIGN/"recoverable_cache_rotation.json", receipt)


def source_path():
    return ROOT/("releases/positive_regime_"+read(CAMPAIGN/"transfer.json")["source_tree_sha256"][:12])


def imports():
    source = source_path()
    sys.path[:0] = [str(source/"src"), str(source/"experiments")]
    importlib.invalidate_caches()
    os.environ["PYTHONPATH"] = str(source/"src")+":"+str(source/"experiments")
    import positive_regime_confirmation as runner
    return runner


def prepare():
    metadata = read(CAMPAIGN/"transfer.json")
    if sha(CAMPAIGN/"source.tar.gz") != metadata["archive_sha256"]:
        raise ValueError("Transferred archive differs")
    if sha(CAMPAIGN/"preregistration.md") != metadata["preregistration_sha256"]:
        raise ValueError("Frozen preregistration differs")
    if sha(__file__) != metadata["driver_sha256"]:
        raise ValueError("Operational driver differs")
    source = source_path()
    source.mkdir(parents=True, exist_ok=True)
    with tarfile.open(CAMPAIGN/"source.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not (source/member.name).resolve().is_relative_to(source.resolve()):
                raise ValueError("Unsafe source archive")
        for member in archive.getmembers():
            dest = source/member.name
            payload = archive.extractfile(member).read()
            if dest.exists():
                if dest.is_symlink() or sha(dest) != hashlib.sha256(payload).hexdigest():
                    raise ValueError("Existing source differs")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
    runner = imports()
    from third_eye.provenance import source_inventory
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.final import load_final_manifest
    if source_inventory(source)["source_tree_sha256"] != metadata["source_tree_sha256"]:
        raise ValueError("Frozen scientific source differs")
    original = read(DEV/"request.json")
    if sha(DEV/"development_review.json") != metadata["development_review_sha256"]:
        raise ValueError("Completed development review differs")
    task = next(t for t in original["reference_tasks"] if t["stream"] == "qwen3-4b/math")
    request = {"declaration": runner.DECLARATION, "source": str(source),
        "source_tree_sha256": metadata["source_tree_sha256"], "output": str(CAMPAIGN),
        "validation": str(CAMPAIGN/"validation/completed.json"),
        "config": task["config"], "manifest": task["manifest"],
        "final_manifest": task["final_manifest"], "model_sources": original["model_sources"],
        "development_review": str(DEV/"development_review.json"),
        "preregistration_sha256": metadata["preregistration_sha256"]}
    for field in ("config", "manifest", "final_manifest", "model_sources", "development_review"):
        request[field+"_sha256"] = sha(request[field])
    for field in ("config", "manifest", "final_manifest"):
        if request[field+"_sha256"] != task[field+"_sha256"]:
            raise ValueError("Inherited frozen input differs")
    write(CAMPAIGN/"request.json", request)
    runner.checked(CAMPAIGN/"request.json", validation=False)
    runner.configured(request, runner.DECLARATION["seeds"][0])
    load_manifest(request["manifest"])
    load_final_manifest(request["final_manifest"], request["manifest"])
    if usage()+8_000_000_000 > LIMIT:
        retire_completed_llama_cache()
    if usage()+8_000_000_000 > LIMIT:
        raise ValueError("Eight GB whole-cohort storage headroom required")
    os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
    from next_round_regimes import verify_models
    checks = verify_models({"reference_tasks": [task]})
    validation = CAMPAIGN/"validation"
    validation.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", str(validation/"fixtures")],
                   cwd=source, check=True)
    subprocess.run([sys.executable, "-m", "ruff", "check", "src", "experiments", "tests"],
                   cwd=source, check=True)
    write(validation/"completed.json", {"passed": True, "model_checks": checks,
        "source_tree_sha256": metadata["source_tree_sha256"],
        "archive_sha256": metadata["archive_sha256"], "request_sha256": sha(CAMPAIGN/"request.json"),
        "preregistration_sha256": metadata["preregistration_sha256"], "driver_sha256": sha(__file__),
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(), "home_bytes": usage()})
    print(json.dumps({"prepared": True, "home_bytes": usage()}), flush=True)


def parent(index):
    runner = imports()
    request = runner.checked(CAMPAIGN/"request.json")
    os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
    if usage()+2_000_000_000 > LIMIT:
        raise ValueError("Insufficient whole-home GPU scratch headroom")
    runner.run_parent(request, index)
    from third_eye.evaluation.final import load_final_manifest
    from third_eye.experiments.scale import compact_checkpoints
    _, final = load_final_manifest(request["final_manifest"], request["manifest"])
    runner.inspect_parent(request, index, final)
    folder = CAMPAIGN/"parents"/str(runner.DECLARATION["seeds"][index])
    compact_checkpoints(folder)
    if usage() > LIMIT:
        raise ValueError("Whole-home policy exceeded after archival")


def timing():
    runner = imports()
    runner.checked(CAMPAIGN/"request.json")
    records = []
    for seed in runner.DECLARATION["seeds"][:2]:
        folder = CAMPAIGN/"parents"/str(seed)
        done = read(folder/"completed.json")
        if not done["passed"] or not read(folder/"lossless_retention.json")["passed"]:
            raise ValueError("Two complete retained operational pilots required")
        records.append({"seed": seed, "elapsed_seconds": done["elapsed_seconds"], "retained_bytes": usage(folder)})
    worst = max(r["retained_bytes"] for r in records)
    projected = usage()+18*worst+2_000_000_000
    receipt = {"passed": projected <= LIMIT, "pilots": records, "projected_home_bytes": projected,
        "remaining_serial_hours_at_slowest_pilot": 18*max(r["elapsed_seconds"] for r in records)/3600,
        "scientific_outcomes_used": False, "scientific_settings_changed": False}
    write(CAMPAIGN/"timing_review.json", receipt)
    if not receipt["passed"]:
        raise ValueError("Projected retained cohort exceeds whole-home quota; preserve completed work")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "parent", "timing", "review"))
    parser.add_argument("--index", type=int, choices=range(20))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Every operation requires a Slurm compute node")
    if args.mode == "prepare":
        prepare()
    elif args.mode == "parent":
        if args.index is None:
            parser.error("parent requires --index")
        parent(args.index)
    elif args.mode == "timing":
        timing()
    else:
        runner = imports()
        runner.review(runner.checked(CAMPAIGN/"request.json"))
