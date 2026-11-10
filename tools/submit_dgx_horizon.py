"""Login-safe dependency submission for the fixed sixteen paired trajectories."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path("/dgxa_home/se23ucse205/third_eye_horizon_20261109")
PRIMARY = Path("/dgxa_home/se23uari167/third_eye_dgx_20261026")
PYTHON = PRIMARY/".venv/bin/python"
if (ROOT/"submissions.json").exists():
    raise RuntimeError("Submission receipt exists; do not duplicate research jobs")
meta = json.loads((ROOT/"transfer.json").read_text())
source = ROOT/("release_"+meta["source_tree_sha256"][:12])
models = ["llama-3.2-3b-instruct", "qwen3-8b", "llama3.1-8b", "qwen3-4b"]
ceilings = {"llama-3.2-3b-instruct": 24, "qwen3-8b": 32, "llama3.1-8b": 32, "qwen3-4b": 24}
receipt = {"source_tree_sha256": meta["source_tree_sha256"], "paired_seeds": 16,
    "worker_gpu_hour_ceiling": 450, "gpu_concurrency": 1, "jobs": [], "vista_sus": 0}


def submit(name, mode, *, gpu=False, model=None, index=None, dependencies=(), hours=None):
    cpu = 16 if gpu else 2 if mode == "archive" else 8
    mem = "64G" if gpu else "8G" if mode == "archive" else "32G"
    duration = "00:30:00" if mode == "compatibility" else str(hours)+":00:00" if gpu else "04:00:00"
    command = mode+(" --model "+model if model else "")+(" --index "+str(index) if index is not None else "")
    lines = ["#!/bin/bash", "#SBATCH --job-name=te-horizon-"+name,
        "#SBATCH --account=student", "#SBATCH --partition="+("gpu_student" if gpu else "cpu_student"),
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task="+str(cpu),
        "#SBATCH --mem="+mem, "#SBATCH --time="+duration, "#SBATCH --chdir="+str(ROOT),
        "#SBATCH --output="+str(ROOT/"logs"/(name+"-%j.out")),
        "#SBATCH --error="+str(ROOT/"logs"/(name+"-%j.err"))]
    if gpu:
        lines += ["#SBATCH --gres=gpu:a100:1"]
    lines += ["set -euo pipefail",
        '[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1',
        "export PYTHONPATH="+str(source/"src")+":"+str(source/"experiments"),
        "export HF_HOME="+str(PRIMARY/"models/hf_cache"),
        "export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false",
        "export OMP_NUM_THREADS="+("4" if gpu else "2")+" OPENBLAS_NUM_THREADS=1",
        "export CUBLAS_WORKSPACE_CONFIG=:4096:8 THIRD_EYE_SYMBOLIC_WORKERS=1",
        "export THIRD_EYE_VERIFIER_WORKERS="+("16" if gpu else str(cpu))]
    if gpu:
        lines += ['[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 1',
                  "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"]
    else:
        lines += ['export CUDA_VISIBLE_DEVICES=""']
    lines += [str(PYTHON)+" "+str(ROOT/"runtime.py")+" "+command]
    path = ROOT/(name+".slurm")
    path.write_text("\n".join(lines)+"\n")
    call = ["sbatch", "--parsable"]
    if dependencies:
        call += ["--dependency=afterok:"+":".join(dependencies)]
    result = subprocess.run(call+[str(path)], cwd=Path.home(), capture_output=True, text=True)
    record = {"name": name, "mode": mode, "model": model, "index": index,
        "dependencies": list(dependencies), "stdout": result.stdout,
        "stderr": result.stderr, "exit_code": result.returncode}
    receipt["jobs"].append(record)
    (ROOT/"submissions.json").write_text(json.dumps(receipt, indent=2)+"\n")
    matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$", result.stdout, re.M)
    if result.returncode or len(matches) != 1:
        raise RuntimeError("Submission failed; inspect partial receipt: "+json.dumps(record))
    record["job_id"] = matches[0]
    (ROOT/"submissions.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"name": name, "job_id": matches[0], "dependencies": list(dependencies)}), flush=True)
    return matches[0]


validation = submit("validate", "validate")
binding = submit("bind", "bind", dependencies=(validation, meta["models_job_id"]))
compat = [submit("compat-"+model, "compatibility", gpu=True, model=model, dependencies=(binding,)) for model in models]
reviews = []
for model in models:
    archived = []
    for index in range(4):
        dependencies = list(compat)
        if model == "qwen3-4b":
            dependencies.append(("35051_" if index < 2 else "35053_")+str(index))
        run = submit(model+"-"+str(index), "run", gpu=True, model=model, index=index,
                     dependencies=dependencies, hours=ceilings[model])
        archived.append(submit("archive-"+model+"-"+str(index), "archive", model=model, index=index,
                               dependencies=(run,)))
    reviews.append(submit("review-"+model, "review", model=model, dependencies=archived))
submit("final", "final", dependencies=reviews)
