"""Metadata-only staging/submission; scientific work runs in the batch scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


VALIDATOR = r'''
import hashlib,json,os,socket,subprocess,sys,tarfile
from pathlib import Path
assert os.environ.get("SLURM_JOB_ID") and "login" not in socket.gethostname()
request=json.loads(Path(sys.argv[1]).read_text())
archive=Path(request["archive"])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==request["archive_sha256"]
dest=Path(request["source"])
dest.mkdir(parents=True,exist_ok=False)
with tarfile.open(archive,"r:gz") as bundle:
    for member in bundle.getmembers():
        target=(dest/member.name).resolve()
        if not target.is_relative_to(dest.resolve()) or not member.isfile():
            raise ValueError("Only regular source files inside release are allowed")
    bundle.extractall(dest)
sys.path[:0]=[str(dest/"src"),str(dest/"experiments")]
from third_eye.provenance import source_inventory
assert source_inventory(dest)["source_tree_sha256"]==request["source_tree_sha256"]
os.environ["PYTHONPATH"]=str(dest/"src")+":"+str(dest/"experiments")
result={"passed":False,"source_tree_sha256":request["source_tree_sha256"],
        "archive_sha256":request["archive_sha256"],"job_id":os.environ["SLURM_JOB_ID"],
        "host":socket.gethostname()}
(Path(request["output"])/"validation").mkdir(parents=True,exist_ok=True)
try:
    subprocess.run([sys.executable,"-m","pytest","-q","--basetemp",
                    str(Path(request["output"])/"validation"/"fixtures")],cwd=dest,check=True)
    subprocess.run([sys.executable,"-m","ruff","check","src","experiments","tests"],
                   cwd=dest,check=True)
    result["passed"]=True
finally:
    Path(request["validation"]).write_text(json.dumps(result,indent=2)+"\n")
'''


def stage(root, site):
    root = Path(root)
    campaign = root/"runs/next_round_20261106"
    payload = read(campaign/"payload.json")
    source = root/("releases/next_round_"+payload["source_tree_sha256"][:12])
    if (campaign/"request.json").exists():
        raise RuntimeError("Campaign already frozen; do not replace its request")
    model_sources = campaign/"model_sources.json"
    mapping = {} if site == "dgx" else {
        "meta-llama/Llama-3.2-3B-Instruct": str(root/"models/mirror_verification/llama3_2_3b_vista.json")}
    write(model_sources, mapping)
    request = {**payload, "source": str(source), "archive": str(campaign/"source_v2.tar.gz"),
               "output": str(campaign), "model_sources": str(model_sources),
               "model_sources_sha256": sha(model_sources),
               "validation": str(campaign/"validation/completed.json"), "reference_tasks": []}
    core = root/"runs/final_empirical_confirmation_43b5c215c6b0/plan.json"
    if site == "vista":
        tasks = read(core)["tasks"]
        for model in payload["declaration"]["models"]:
            task = next(t for t in tasks if t["stream"] == model+"/math")
            task = dict(task)
            for key in ("config", "manifest", "final_manifest"):
                task[key+"_sha256"] = sha(task[key])
            request["reference_tasks"].append(task)
    write(campaign/"request.json", request)
    (campaign/"validation").mkdir()
    (campaign/"logs").mkdir()
    (campaign/"validate.py").write_text(VALIDATOR)
    bootstrap = root/"releases/main_8282f1d02158"

    def script(name, partition, hours, body, cpus=16, array=None):
        account = "IRI23021" if site == "vista" else "student"
        lines = ["#!/bin/bash", "#SBATCH --job-name=te-next-"+name,
                 "#SBATCH --partition="+partition, "#SBATCH --account="+account,
                 "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
                 "#SBATCH --cpus-per-task="+str(cpus),
                 "#SBATCH --time="+str(hours)+":00:00",
                 "#SBATCH --output="+str(campaign/"logs"/(name+"-%A_%a.out")),
                 "#SBATCH --error="+str(campaign/"logs"/(name+"-%A_%a.err"))]
        if site == "dgx":
            lines += ["#SBATCH --mem=16G"]
        if array:
            lines += ["#SBATCH --array="+array]
        lines += ["set -euo pipefail",
                  '[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1',
                  "export THIRD_EYE_SOURCE_ROOT="+str(bootstrap if name=="validate" else source),
                  "export THIRD_EYE_VENV="+str(root/".venv")]
        if site == "vista":
            lines += ['source "$THIRD_EYE_SOURCE_ROOT/experiments/jobs/bounded_evidence_environment.sh"']
        else:
            lines += ['export PYTHONPATH="$THIRD_EYE_SOURCE_ROOT/src:$THIRD_EYE_SOURCE_ROOT/experiments"',
                      "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1",
                      'cd "$THIRD_EYE_SOURCE_ROOT"']
        lines += ["export CUDA_VISIBLE_DEVICES="+("0" if name=="pilots" else '""'),
                  "export HF_HOME="+str(root/"models/hf_cache"),
                  "export THIRD_EYE_MODEL_SOURCES="+str(model_sources),
                  "export TOKENIZERS_PARALLELISM=false THIRD_EYE_SYMBOLIC_WORKERS=1",
                  body]
        (campaign/(name+".slurm")).write_text("\n".join(lines)+"\n")

    python = str(root/".venv/bin/python")
    request_arg = " --request "+str(campaign/"request.json")
    script("validate", "gg" if site=="vista" else "cpu_student", 1,
           python+" "+str(campaign/"validate.py")+" "+str(campaign/"request.json"), cpus=4)
    if site == "vista":
        script("prepare", "gg", 1,
               "export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0\n"+python+
               " experiments/next_round_regimes.py prepare"+request_arg, cpus=8)
        plans = [core, root/"runs/smollm16_090fc9a12413/plan.json",
                 root/"runs/llama8_vista_20261031_8282f1d02158/llama3.1-8b/plan.json"]
        script("pilots", "gh", 6,
               python+" experiments/next_round_regimes.py pilot"+request_arg+
               ' --index "$SLURM_ARRAY_TASK_ID"', array="0-1%2")
    else:
        plans = [root/"runs/scale_8777e39cac29/qwen3-8b/plan.json"]
    script("saved", "gg" if site=="vista" else "cpu_student", 4,
           python+" experiments/next_round_saved.py"+
           "".join(" --plan "+str(p) for p in plans)+" --output "+str(campaign/("saved_"+site)), cpus=4)
    write(campaign/"ledger.json", {"site":site, "additional_su_cap":200,
          "first_stage_su_cap":55, "reservations":[], "charged_and_reserved_sus":0,
          "note":"Full walltime reserved, including CPU minimums and failed jobs; no automatic confirmation"})
    print(json.dumps({"staged":str(campaign), "source_tree_sha256":payload["source_tree_sha256"],
                      "request_sha256":sha(campaign/"request.json")}))


def submit(root, site):
    campaign = Path(root)/"runs/next_round_20261106"
    ledger_path = campaign/"ledger.json"
    if read(ledger_path)["reservations"]:
        raise RuntimeError("Already submitted or partially submitted; inspect ledger before proceeding")

    def one(name, sus, dependency=None):
        ledger = read(ledger_path)
        total = ledger["charged_and_reserved_sus"]+sus
        if total > min(ledger["additional_su_cap"],ledger["first_stage_su_cap"]):
            raise RuntimeError("Budget reservation denied")
        record = {"name":name,"reserved_sus":sus,"state":"SUBMITTING"}
        ledger["reservations"].append(record)
        ledger["charged_and_reserved_sus"] = total
        write(ledger_path,ledger)
        cmd = ["sbatch","--parsable"]
        if dependency:
            cmd += ["--dependency=afterok:"+dependency]
        cmd += [str(campaign/(name+".slurm"))]
        outcome = subprocess.run(cmd, text=True,capture_output=True)
        record.update({"stdout":outcome.stdout,"stderr":outcome.stderr,"exit_code":outcome.returncode})
        matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$",outcome.stdout,re.M)
        if outcome.returncode or len(matches)!=1:
            record["state"]="SUBMISSION_REQUIRES_REVIEW"
            write(ledger_path,ledger)
            raise RuntimeError("Submission failed; reservation retained: "+json.dumps(record))
        record.update({"state":"SUBMITTED","job_id":matches[0],"dependency":dependency})
        write(ledger_path,ledger)
        print(json.dumps(record),flush=True)
        return matches[0]
    validate = one("validate",.33 if site=="vista" else 0)
    if site == "vista":
        prepare = one("prepare",.33,validate)
        one("saved",1.32,validate)
        one("pilots",12,validate+":"+prepare)
    else:
        one("saved",0,validate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("stage","submit"))
    parser.add_argument("--root",required=True)
    parser.add_argument("--site",choices=("vista","dgx"),required=True)
    args = parser.parse_args()
    {"stage":stage,"submit":submit}[args.action](args.root,args.site)
