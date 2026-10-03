#!/bin/bash
#SBATCH --job-name=third-eye-study
#SBATCH --partition=gpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=third_eye_study_%A_%a.out
# Specify an allowed GPU/MIG resource at submission; never assume 5GB fits 4B.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
[[ "$(hostname)" != *login* ]] || exit 1
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'GPU allocation missing'; exit 1; }
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
export OMP_NUM_THREADS="${THIRD_EYE_OMP_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${SLURM_TMPDIR:-$PWD/.tmp/${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
nvidia-smi
python experiments/task.py --plan "${THIRD_EYE_PLAN:?Set the generated study plan}" \
  --stage "${THIRD_EYE_STAGE:?Set the study stage}" --index "${SLURM_ARRAY_TASK_ID:-0}"
