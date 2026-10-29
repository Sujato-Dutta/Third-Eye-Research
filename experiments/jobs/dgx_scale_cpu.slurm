#!/bin/bash
#SBATCH --job-name=third-eye-scale-cpu
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]]
root="$HOME/third_eye_dgx_20261026"
cd "${THIRD_EYE_SOURCE_ROOT:?}"
export PYTHONPATH="$PWD/src:$PWD/experiments" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HF_HOME="$root/models/hf_cache" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1
arguments=("${THIRD_EYE_SCALE_ACTION:?}" --request "${THIRD_EYE_SCALE_REQUEST:?}")
if [[ -n "${THIRD_EYE_SCALE_MODEL:-}" ]]; then arguments+=(--model "$THIRD_EYE_SCALE_MODEL"); fi
"$root/.venv/bin/python" experiments/scale_confirmation.py "${arguments[@]}"
