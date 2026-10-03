#!/bin/bash
#SBATCH --job-name=third-eye-study
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=third_eye_vista_study_%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/experiments/jobs/vista_environment.sh"
nvidia-smi
python experiments/task.py --plan "${THIRD_EYE_PLAN:?}" \
    --stage "${THIRD_EYE_STAGE:?}" --index "${SLURM_ARRAY_TASK_ID:-0}"

