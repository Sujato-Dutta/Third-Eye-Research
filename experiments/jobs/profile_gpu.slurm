#!/bin/bash
#SBATCH --job-name=third-eye-gpu-profile
#SBATCH --partition=gpu_student
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=third_eye_gpu_profile_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(hostname)" != *login* ]] || exit 1
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
python experiments/profile_gpu.py
