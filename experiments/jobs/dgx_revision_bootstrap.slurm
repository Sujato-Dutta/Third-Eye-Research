#!/bin/bash
#SBATCH --job-name=third-eye-revision-setup
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=third_eye_dgx_setup_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]]
root="$HOME/third_eye_dgx_20261026"
cd "$root"
before=$(du -x -B1 -s "$HOME" | cut -f1)
(( before + 12000000000 <= 50000000000 )) || { echo 'Insufficient 50GB policy headroom'; exit 1; }
export PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TMPDIR="$root/tmp/setup_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
[[ ! -e "$root/.venv" ]] || { echo 'Preserve existing environment; inspect before retry'; exit 1; }
python3 -m venv "$root/.venv"
"$root/.venv/bin/python" -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu126
"$root/.venv/bin/python" -m pip install -r "$root/requirements-evaluation.txt"
"$root/.venv/bin/python" -m pip freeze > "$root/environment_freeze.txt"
"$root/.venv/bin/python" - <<'PY'
import importlib.metadata as md, json, os, pathlib, socket
packages=['torch','transformers','peft','accelerate','numpy','safetensors','pytest','ruff','datasets','math-verify','matplotlib']
record={'passed':True,'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'versions':{p:md.version(p) for p in packages}}
path=pathlib.Path.home()/'third_eye_dgx_20261026'/'environment_completed.json'
path.write_text(json.dumps(record,indent=2))
print(json.dumps(record))
PY
du -x -B1 -s "$HOME" "$root/.venv"
