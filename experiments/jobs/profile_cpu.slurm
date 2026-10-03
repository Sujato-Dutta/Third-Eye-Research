#!/bin/bash
#SBATCH --job-name=third-eye-cpu-profile
#SBATCH --partition=cpu_student
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=third_eye_cpu_profile_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(hostname)" != *login* ]] || exit 1
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q
python experiments/profile_verification.py
