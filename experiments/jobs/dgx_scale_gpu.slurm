#!/bin/bash
#SBATCH --job-name=third-eye-scale-gpu
#SBATCH --partition=gpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
root="$HOME/third_eye_dgx_20261026"
cd "${THIRD_EYE_SOURCE_ROOT:?}"
export PYTHONPATH="$PWD/src:$PWD/experiments" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$root/models/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export THIRD_EYE_MODEL_SOURCES="${THIRD_EYE_SCALE_OUTPUT:?}/${THIRD_EYE_SCALE_MODEL:?}/model_sources.json"
export THIRD_EYE_SANDBOX_CONFIG="$root/sandbox.json"
arguments=("${THIRD_EYE_SCALE_ACTION:?}" --request "${THIRD_EYE_SCALE_REQUEST:?}" --model "$THIRD_EYE_SCALE_MODEL")
if [[ "$THIRD_EYE_SCALE_ACTION" == run ]]; then arguments+=(--index "${SLURM_ARRAY_TASK_ID:?}"); fi
"$root/.venv/bin/python" experiments/scale_confirmation.py "${arguments[@]}"
