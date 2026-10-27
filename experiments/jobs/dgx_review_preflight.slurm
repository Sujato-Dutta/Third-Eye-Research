#!/bin/bash
#SBATCH --job-name=third-eye-review-compatibility
#SBATCH --partition=gpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=third_eye_dgx_preflight_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
root="$HOME/third_eye_dgx_20261026"
cd "${THIRD_EYE_SOURCE_ROOT:?}"
export PYTHONPATH="$PWD/src:$PWD/experiments" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1
export HF_HOME="$root/models/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export THIRD_EYE_MODEL_SOURCES="$root/model_sources.json" THIRD_EYE_SANDBOX_CONFIG="$root/sandbox.json"
"$root/.venv/bin/python" experiments/review_controls.py preflight --request "${THIRD_EYE_REVIEW_REQUEST:?}"
