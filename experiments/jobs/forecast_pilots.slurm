#!/bin/bash
#SBATCH --job-name=third-eye-fresh-pilots
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --array=0-7%8
#SBATCH --account=IRI23021
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
source_root="${THIRD_EYE_SOURCE_ROOT:?}"
venv_root="${THIRD_EYE_VENV:?}"
module purge
module load gcc cuda python3
python_binary=$(readlink -f "$venv_root/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$python_binary")")/lib:${LD_LIBRARY_PATH:-}"
cd "$source_root"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$source_root/src:$source_root/experiments"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export THIRD_EYE_VERIFIER=third_eye.evaluation.benchmarks:make_verifier
export THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1
"$venv_root/bin/python" experiments/forecasting_pilots.py run --plan "${THIRD_EYE_PILOT_PLAN:?}" --index "$((SLURM_ARRAY_TASK_ID + ${THIRD_EYE_INDEX_BASE:-0}))"
