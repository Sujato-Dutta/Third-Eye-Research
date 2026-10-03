#!/bin/bash
#SBATCH --job-name=third-eye-controller
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=third_eye_controller_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(hostname)" != *login* ]] || exit 1
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
python experiments/continue_study.py \
    --campaign "${THIRD_EYE_CAMPAIGN:?}" \
    --output "${THIRD_EYE_CAMPAIGN_OUTPUT:?}" \
    --gres "${THIRD_EYE_GRES:-gpu:a100_4g.20gb:1}"
