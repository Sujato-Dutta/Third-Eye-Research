#!/bin/bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
source_root="${THIRD_EYE_SOURCE_ROOT:?}"
venv_root="${THIRD_EYE_VENV:?}"
module purge
module load gcc cuda python3
python_binary=$(readlink -f "$venv_root/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$source_root/src:$source_root/experiments" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1
cd "$source_root"
