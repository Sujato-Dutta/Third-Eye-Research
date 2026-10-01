#!/bin/bash
#SBATCH --job-name=third-eye-data
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_data_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
[[ "$(hostname)" != *login* ]] || exit 1
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
python experiments/prepare_data.py --task math --output "${THIRD_EYE_DATA_ROOT:-data/processed/v1}/math"
python experiments/prepare_data.py --task code --output "${THIRD_EYE_DATA_ROOT:-data/processed/v1}/code"
