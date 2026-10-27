#!/bin/bash
#SBATCH --job-name=third-eye-controls-analysis
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_dgx_controls_analysis_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]]
root="$HOME/third_eye_dgx_20261026"
cd "${THIRD_EYE_SOURCE_ROOT:?}"
export PYTHONPATH="$PWD/src:$PWD/experiments" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$root/.venv/bin/python" experiments/review_controls.py review --request "${THIRD_EYE_REVIEW_REQUEST:?}"
