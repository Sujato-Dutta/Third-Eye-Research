#!/bin/bash
#SBATCH --job-name=third-eye-audit
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_vista_audit_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module load gcc cuda python3
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export THIRD_EYE_MODEL_SOURCES="$PWD/experiments/model_sources_vista.json"
export THIRD_EYE_SANDBOX_CONFIG="$PWD/experiments/sandbox_vista.json"
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
python cluster/vista_audit.py
