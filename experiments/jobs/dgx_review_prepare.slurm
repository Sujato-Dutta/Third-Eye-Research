#!/bin/bash
#SBATCH --job-name=third-eye-review-import
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=third_eye_dgx_import_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]]
root="$HOME/third_eye_dgx_20261026"
cd "${THIRD_EYE_SOURCE_ROOT:?}"
export PYTHONPATH="$PWD/src:$PWD/experiments" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
"$root/.venv/bin/python" tools/dgx_prepare_review.py --root "$root"
