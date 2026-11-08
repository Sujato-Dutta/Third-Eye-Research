"""Login-safe Slurm staging; no research execution outside allocations."""
import json
from pathlib import Path
import re
import subprocess


ROOT = Path("/dgxa_home/se23uari167/third_eye_dgx_20261026")
CAMPAIGN = ROOT/"runs/next_round_dgx_20261108"
if (CAMPAIGN/"submissions.json").exists():
    raise RuntimeError("Campaign already submitted; inspect existing receipts")
(CAMPAIGN/"logs").mkdir(exist_ok=True)
metadata = json.loads((CAMPAIGN/"transfer.json").read_text())
source = ROOT/("releases/next_round_"+metadata["source_tree_sha256"][:12])
driver = CAMPAIGN/"runtime.py"
python = ROOT/".venv/bin/python"


def script(name,body,gpu=False,array=None):
    lines = ["#!/bin/bash","#SBATCH --job-name=te-dgx-next-"+name,
             "#SBATCH --account=student",
             "#SBATCH --partition="+("gpu_student" if gpu else "cpu_student"),
             "#SBATCH --nodes=1","#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task="+("16" if gpu else "8"),
             "#SBATCH --mem="+("64G" if gpu else "32G"),
             "#SBATCH --time="+("1-00:00:00" if gpu else "04:00:00"),
             "#SBATCH --chdir="+str(ROOT),
             "#SBATCH --output="+str(CAMPAIGN/"logs"/(name+"-%A_%a.out")),
             "#SBATCH --error="+str(CAMPAIGN/"logs"/(name+"-%A_%a.err"))]
    if gpu:
        lines += ["#SBATCH --gres=gpu:a100:1"]
    if array:
        lines += ["#SBATCH --array="+array]
    lines += ["set -euo pipefail",
              '[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1',
              "export PYTHONPATH="+str(source/"src")+":"+str(source/"experiments"),
              "export HF_HOME="+str(ROOT/"models/hf_cache"),
              "export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false",
              "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1",
              "export CUBLAS_WORKSPACE_CONFIG=:4096:8 THIRD_EYE_SYMBOLIC_WORKERS=1",
              "export THIRD_EYE_VERIFIER_WORKERS="+("16" if gpu else "4")]
    if gpu:
        # Keep the scheduler-provided mask; never force a physical GPU index.
        lines += ['[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 1',
                  "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"]
    else:
        lines += ['export CUDA_VISIBLE_DEVICES=""']
    lines += [str(python)+" "+str(driver)+" "+body]
    path = CAMPAIGN/(name+".slurm")
    path.write_text("\n".join(lines)+"\n")
    return path


paths = {
    "prepare":script("prepare","prepare"),
    "initial":script("initial",'pilot --block initial --index "$SLURM_ARRAY_TASK_ID"',True,"0-1%1"),
    "remaining":script("remaining",'pilot --block remaining --index "$SLURM_ARRAY_TASK_ID"',True,"0-5%1"),
    "review":script("review","review"),
}
receipts = {"vista_sus_reserved":0,"gpu_concurrency":1,"automatic_confirmation":False,
            "source_tree_sha256":metadata["source_tree_sha256"],"jobs":[]}


def submit(name,dependency=None):
    cmd = ["sbatch","--parsable"]
    if dependency:
        cmd += ["--dependency=afterok:"+dependency]
    cmd += [str(paths[name])]
    result = subprocess.run(cmd,cwd=Path.home(),capture_output=True,text=True)
    record = {"name":name,"stdout":result.stdout,"stderr":result.stderr,
              "exit_code":result.returncode,"dependency":dependency}
    receipts["jobs"].append(record)
    (CAMPAIGN/"submissions.json").write_text(json.dumps(receipts,indent=2))
    matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$",result.stdout,re.M)
    if result.returncode or len(matches)!=1:
        raise RuntimeError("Submission failed; preserve partial receipts: "+json.dumps(record))
    record["job_id"] = matches[0]
    (CAMPAIGN/"submissions.json").write_text(json.dumps(receipts,indent=2))
    print(json.dumps(record),flush=True)
    return matches[0]


prepare = submit("prepare")
initial = submit("initial",prepare)
remaining = submit("remaining",initial)
submit("review",initial+":"+remaining)
