"""Scheduler-only submission: one A100, validated dependencies, fixed cohort."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path("/dgxa_home/se23uari167/third_eye_dgx_20261026")
CAMPAIGN = ROOT/"runs/positive_regime_20261109"
if (CAMPAIGN/"submissions.json").exists():
    raise RuntimeError("Submission receipt exists; inspect rather than duplicate jobs")
metadata = json.loads((CAMPAIGN/"transfer.json").read_text())
source = ROOT/("releases/positive_regime_"+metadata["source_tree_sha256"][:12])
(CAMPAIGN/"logs").mkdir(exist_ok=True)


def script(name, command, gpu=False, array=None):
    lines = ["#!/bin/bash", "#SBATCH --job-name=te-positive-"+name,
        "#SBATCH --account=student", "#SBATCH --partition="+("gpu_student" if gpu else "cpu_student"),
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task="+("16" if gpu else "8"),
        "#SBATCH --mem="+("64G" if gpu else "32G"),
        "#SBATCH --time="+("2-00:00:00" if gpu else "04:00:00"),
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
        lines += ['[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 1',
                  "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"]
    else:
        lines += ['export CUDA_VISIBLE_DEVICES=""']
    lines += [str(ROOT/".venv/bin/python")+" "+str(CAMPAIGN/"runtime.py")+" "+command]
    path = CAMPAIGN/(name+".slurm")
    path.write_text("\n".join(lines)+"\n")
    return path


paths = {
    "prepare": script("prepare", "prepare"),
    "initial": script("initial", 'parent --index "$SLURM_ARRAY_TASK_ID"', True, "0-1%1"),
    "timing": script("timing", "timing"),
    "remaining": script("remaining", 'parent --index "$SLURM_ARRAY_TASK_ID"', True, "2-19%1"),
    "review": script("review", "review"),
}
receipts = {"vista_sus": 0, "gpu_concurrency": 1, "parents": 20,
            "source_tree_sha256": metadata["source_tree_sha256"], "jobs": []}


def submit(name, dependency=None):
    command = ["sbatch", "--parsable"]
    if dependency:
        command += ["--dependency=afterok:"+dependency]
    command += [str(paths[name])]
    result = subprocess.run(command, cwd=Path.home(), capture_output=True, text=True)
    record = {"name": name, "dependency": dependency, "stdout": result.stdout,
              "stderr": result.stderr, "exit_code": result.returncode}
    receipts["jobs"].append(record)
    (CAMPAIGN/"submissions.json").write_text(json.dumps(receipts, indent=2)+"\n")
    matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$", result.stdout, re.M)
    if result.returncode or len(matches) != 1:
        raise RuntimeError("Submission failed; inspect partial receipt: "+json.dumps(record))
    record["job_id"] = matches[0]
    (CAMPAIGN/"submissions.json").write_text(json.dumps(receipts, indent=2)+"\n")
    print(json.dumps(record), flush=True)
    return matches[0]


prepare = submit("prepare")
initial = submit("initial", prepare)
timing = submit("timing", initial)
remaining = submit("remaining", timing)
submit("review", initial+":"+remaining)
