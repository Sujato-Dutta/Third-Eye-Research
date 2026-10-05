#!/bin/bash
#SBATCH --job-name=third-eye-paired-retry-control
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=third_eye_paired_retry_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module purge
module load gcc cuda python3
source "${THIRD_EYE_VENV:?}/bin/activate"
third_eye_python_binary=$(readlink -f "$THIRD_EYE_VENV/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$third_eye_python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export THIRD_EYE_MODEL_SOURCES="${THIRD_EYE_ORIGINAL_ROOT:?}/experiments/model_sources_vista.json"
export THIRD_EYE_SANDBOX_CONFIG="$THIRD_EYE_ORIGINAL_ROOT/experiments/sandbox_vista.json"
export HF_HOME="${THIRD_EYE_HF_HOME:?}" CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1
tasks=(math code)
index="${SLURM_ARRAY_TASK_ID:?}"
[[ "$index" =~ ^[01]$ ]] || exit 1
task="${tasks[$index]}"
python tools/paired_retry_control.py \
  --config runs/campaign_a1_20261003/configs/llama3_2_3b_1042.json \
  --manifest "data/processed/a1/${task}/selection.json" \
  --reference-run "runs/campaign_a1_20261003/pilots/llama3_2_3b_${task}_1042" \
  --output "runs/retry_diagnostics/${SLURM_ARRAY_JOB_ID}/llama3_2_3b_${task}_paired.json"
