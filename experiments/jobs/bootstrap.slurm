#!/bin/bash
#SBATCH --job-name=third-eye-setup
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_setup_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
[[ "$(hostname)" != *login* ]] || { echo 'Refusing execution on login node'; exit 1; }
export TMPDIR="${SLURM_TMPDIR:-$PWD/.tmp/${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
export PIP_NO_CACHE_DIR=1
# Set the CUDA wheel index to match the installed DGX driver.
python3 -m venv "${THIRD_EYE_VENV:-.venv}"
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
python -m pip install torch==2.8.0 --index-url "${THIRD_EYE_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
python -m pip install -r requirements-research.txt
python -m pip freeze > "runs_dependencies_${SLURM_JOB_ID}.txt"
