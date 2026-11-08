"""Allocated DGX relocation: preserve frozen science and all original results."""
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
CAMPAIGN = ROOT/"runs/next_round_dgx_20261108"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def usage():
    result = subprocess.run(["du","-x","-B1","-s",str(Path.home())],
                            text=True,capture_output=True,check=True)
    return int(result.stdout.split()[0])


def unpack(archive,dest,expected):
    if sha(archive) != expected:
        raise ValueError("Transferred archive differs")
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,"r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not (dest/member.name).resolve().is_relative_to(dest.resolve()):
                raise ValueError("Unsafe archive path or file type")
        for member in bundle.getmembers():
            target = dest/member.name
            payload = bundle.extractfile(member).read()
            if target.exists():
                if target.is_symlink() or sha(target) != hashlib.sha256(payload).hexdigest():
                    raise ValueError("Previously staged file differs: "+str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(payload)


def retire_completed_scale_cache():
    """Only if necessary: recoverable Qwen-8B base weights, never results."""
    active = subprocess.run(["squeue","-h","-u",str(os.getuid()),"-o","%P|%T"],
                            text=True,capture_output=True,check=True).stdout
    if any("gpu" in row and "RUNNING" in row for row in active.splitlines()):
        raise RuntimeError("Do not retire caches while this account runs a GPU job")
    for task,start in (("code",2026110200),("math",2026110100)):
        for seed in range(start,start+4):
            done = ROOT/f"runs/scale_8777e39cac29/qwen3-8b/parents/{task}/{seed}/completed.json"
            if not read(done)["passed"]:
                raise ValueError("Completed scale cohort required before rotating its base cache")
    proof_path = ROOT/"runs/scale_8777e39cac29/qwen3-8b/model_source.json"
    proof = read(proof_path)
    if proof.get("official") != "Qwen/Qwen3-8B" or not proof.get("verified"):
        raise ValueError("Completed scale-model source proof required")
    cache = Path(proof["runtime_directory"])
    if cache.resolve() != (ROOT/"models/scale/qwen3-8b").resolve():
        raise ValueError("Unexpected scale-model cache directory")
    if cache.is_symlink() or not cache.resolve().is_relative_to((ROOT/"models").resolve()):
        raise ValueError("Unsafe recoverable-cache target")
    files = []
    for path in sorted(cache.rglob("*.safetensors")):
        target = path.resolve()
        if not target.is_relative_to(cache.resolve()) or target.stat().st_uid != os.getuid():
            raise ValueError("Weight-cache target escapes allowlist or ownership")
        expected = proof["verified_weight_sha256"].get(path.name)
        actual = sha(target)
        if actual != expected:
            raise ValueError("Refuse retirement of an unexpected pretrained weight")
        files.append({"path":str(path),"target":str(target),"sha256":actual,
                      "bytes":target.stat().st_size})
    receipt = {"before_home_bytes":usage(),"files":files,
               "preserved":"all research results, adapters, metadata and environments",
               "job_id":os.environ["SLURM_JOB_ID"]}
    write(CAMPAIGN/"recoverable_cache_rotation.json",receipt)
    for record in files:
        path,target = Path(record["path"]),Path(record["target"])
        if path.is_symlink():
            path.unlink()
        if target.exists():
            target.unlink()
    receipt.update({"after_home_bytes":usage(),"passed":True})
    write(CAMPAIGN/"recoverable_cache_rotation.json",receipt)


def prepare():
    metadata = read(CAMPAIGN/"transfer.json")
    source = ROOT/("releases/next_round_"+metadata["source_tree_sha256"][:12])
    unpack(CAMPAIGN/"source.tar.gz",source,metadata["archive_sha256"])
    imported = CAMPAIGN/"imported"
    unpack(CAMPAIGN/"inputs.tar.gz",imported,metadata["inputs_archive_sha256"])
    # Move only these newly staged input files; original campaigns are immutable.
    for name in ("inputs","configs"):
        staged,installed = imported/name,CAMPAIGN/name
        if installed.exists():
            for path in staged.iterdir():
                if not path.is_file() or sha(path) != sha(installed/path.name):
                    raise ValueError("Existing staged input differs")
        else:
            staged.rename(installed)
    sys.path[:0] = [str(source/"src"),str(source/"experiments")]
    os.environ["PYTHONPATH"] = str(source/"src")+":"+str(source/"experiments")
    # Startup PYTHONPATH can cache a nonexistent release before extraction.
    importlib.invalidate_caches()
    from third_eye.provenance import source_inventory
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.final import load_final_manifest
    assert source_inventory(source)["source_tree_sha256"] == metadata["source_tree_sha256"]
    original = read(imported/"vista_request.json")
    prepared = read(CAMPAIGN/"inputs/completed.json")
    assert prepared["passed"]
    assert sha(CAMPAIGN/"inputs/selection.json") == prepared["selection_sha256"]
    assert sha(CAMPAIGN/"inputs/final.json") == prepared["final_sha256"]
    load_manifest(CAMPAIGN/"inputs/selection.json")
    load_final_manifest(CAMPAIGN/"inputs/final.json",CAMPAIGN/"inputs/selection.json")
    registry = read(ROOT/"imported/registry.json")
    assert sha(ROOT/"imported/registry.json") == read(ROOT/"prepared.json")["registry_sha256"]
    models = registry["models"]
    folders = {name: ROOT/"models/hf_cache/hub"/("models--"+name.replace("/","--"))/
               "snapshots"/m["revision"] for name,m in models.items()}
    needed = sum(item["bytes"] for name,m in models.items() for filename,item in m["files"].items()
                 if not (folders[name]/filename).is_file())
    before = usage()
    write(CAMPAIGN/"storage_inventory.json",{"home_bytes":before,"missing_model_bytes":needed,
          "limit_bytes":50_000_000_000,"output_headroom_bytes":4_000_000_000})
    if before+needed+4_000_000_000 > 50_000_000_000:
        retire_completed_scale_cache()
    if usage()+needed+4_000_000_000 > 50_000_000_000:
        raise ValueError("Pinned models and output headroom exceed the 50GB policy")
    from huggingface_hub import hf_hub_download
    mappings,verified = {},{}
    for name,m in models.items():
        folder = folders[name]
        folder.mkdir(parents=True,exist_ok=True)
        for filename,item in m["files"].items():
            target = folder/filename
            if not target.exists():
                staged = ROOT/"transfers/model_metadata"/name.replace("/","--")/filename
                if staged.is_file() and not filename.endswith(".safetensors"):
                    import shutil
                    shutil.copyfile(staged,target)
                else:
                    origin = ((m.get("source_proof") or {}).get("metadata_sources",{}).get(filename,{})
                              if not filename.endswith(".safetensors") else {})
                    hf_hub_download(origin.get("repository",m["source_repository"]),filename,
                        revision=origin.get("revision",m["source_revision"]),local_dir=folder,token=False)
            if target.stat().st_size != item["bytes"] or sha(target) != item["sha256"]:
                raise ValueError("Pinned model file differs: "+str(target))
            print(json.dumps({"verified_model_file":name+"/"+filename}),flush=True)
        if m["source_proof"]:
            proof = {k:v for k,v in m["source_proof"].items() if k!="proof_sha256"}
            proof["runtime_directory"] = str(folder)
            path = CAMPAIGN/(name.replace("/","--")+"_proof.json")
            write(path,proof)
            mappings[name] = str(path)
        verified[name] = {"revision":m["revision"],"file_hashes":m["files"],"path":str(folder)}
    modelmap = CAMPAIGN/"model_sources.json"
    write(modelmap,mappings)
    request = {**original,"source":str(source),"archive":str(CAMPAIGN/"source.tar.gz"),
               "output":str(CAMPAIGN),"validation":str(CAMPAIGN/"validation/completed.json"),
               "model_sources":str(modelmap),"model_sources_sha256":sha(modelmap),
               "reference_tasks":[]}
    for task in original["reference_tasks"]:
        task = dict(task)
        config = CAMPAIGN/"configs"/Path(task["config"]).name
        assert sha(config) == task["config_sha256"]
        task.update({"config":str(config),"manifest":str(CAMPAIGN/"inputs/selection.json"),
                     "final_manifest":str(CAMPAIGN/"inputs/final.json")})
        for field in ("config","manifest","final_manifest"):
            task[field+"_sha256"] = sha(task[field])
        request["reference_tasks"].append(task)
    write(CAMPAIGN/"request.json",request)
    os.environ["THIRD_EYE_MODEL_SOURCES"] = str(modelmap)
    import next_round_regimes as runner
    runner.checked(CAMPAIGN/"request.json",validation=False)
    checks = runner.verify_models(request)
    (CAMPAIGN/"validation").mkdir(exist_ok=True)
    subprocess.run([sys.executable,"-m","pytest","-q","--basetemp",str(CAMPAIGN/"validation/fixtures")],
                   cwd=source,check=True)
    subprocess.run([sys.executable,"-m","ruff","check","src","experiments","tests"],cwd=source,check=True)
    write(CAMPAIGN/"validation/completed.json",{
        "passed":True,"source_tree_sha256":metadata["source_tree_sha256"],
        "archive_sha256":metadata["archive_sha256"],"driver_sha256":sha(__file__),
        "job_id":os.environ["SLURM_JOB_ID"],"host":socket.gethostname()})
    write(CAMPAIGN/"migration_completed.json",{"passed":True,"model_checks":checks,
        "models":verified,"home_bytes":usage(),"input_archive_sha256":metadata["inputs_archive_sha256"],
        "original_request_sha256":sha(imported/"vista_request.json"),"request_sha256":sha(CAMPAIGN/"request.json"),
        "vista_sus_consumed":0,"gpu_concurrency":1,"scientific_settings_changed":False,
        "note":"A100 timing is not a Vista SU calibration; native/direct pairs remain within DGX"})
    print(json.dumps({"migration_prepared":True,"home_bytes":usage()}),flush=True)


def pilot(block,index):
    request = read(CAMPAIGN/"request.json")
    source = Path(request["source"])
    sys.path[:0] = [str(source/"src"),str(source/"experiments")]
    import next_round_regimes as runner
    from third_eye.experiments.scale import compact_checkpoints
    if block == "remaining":
        for model in runner.DECLARATION["models"]:
            seed = runner.DECLARATION["development_seeds"][model][0]
            done = read(CAMPAIGN/f"development/{model}/{seed}/completed.json")
            if not done["passed"] or not done["development_complete"]:
                raise ValueError("Complete initial pilots required before remaining parents")
    parent_index = 0 if block=="initial" else 1+index//2
    model_index = index if block=="initial" else index%2
    runner.checked(CAMPAIGN/"request.json",validation=True)
    os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
    if usage()+2_000_000_000 > 50_000_000_000:
        raise ValueError("Insufficient 50GB-policy headroom before GPU parent")
    runner.pilot(request,model_index,parent_index)
    model = runner.DECLARATION["models"][model_index]
    seed = runner.DECLARATION["development_seeds"][model][parent_index]
    compact_checkpoints(CAMPAIGN/f"development/{model}/{seed}")
    if usage()>50_000_000_000:
        raise ValueError("Home policy exceeded after verified compaction")


def review():
    request = read(CAMPAIGN/"request.json")
    sys.path[:0] = [request["source"]+"/src",request["source"]+"/experiments"]
    from next_round_regimes import DECLARATION
    from third_eye.experiments.regime_revision import select_regime
    records,parents = [],[]
    for model in DECLARATION["models"]:
        for seed in DECLARATION["development_seeds"][model]:
            folder = CAMPAIGN/f"development/{model}/{seed}"
            done = read(folder/"completed.json")
            if not done["passed"] or not done["development_complete"] or not read(folder/"lossless_retention.json")["passed"]:
                raise ValueError("Complete retained development cohort required")
            records.extend(done["records"])
            parents.append({"model":model,"seed":seed,"elapsed_seconds":done["elapsed_seconds"],
                            "future_timing_complete":done["future_timing_complete"]})
    choice = select_regime(records)
    write(CAMPAIGN/"development_review.json",{"passed":True,"parents":parents,"records":records,
        "eligible_regime":choice,"decision":"FREEZE_CONFIRMATION_DESIGN" if choice else "STOP_NO_POSITIVE_REGIME",
        "final_outcomes_generated":False,"forecaster_fits":0,"vista_sus_consumed":0,
        "automatic_confirmation":False,"request_sha256":sha(CAMPAIGN/"request.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","pilot","review"))
    parser.add_argument("--block",choices=("initial","remaining"))
    parser.add_argument("--index",type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("All execution requires a Slurm compute allocation")
    {"prepare":prepare,"pilot":lambda:pilot(args.block,args.index),"review":review}[args.mode]()
