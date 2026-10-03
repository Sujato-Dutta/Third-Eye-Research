#!/bin/bash
#SBATCH --job-name=third-eye-a1-setup
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_a1_setup_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module load gcc cuda python3
source "${THIRD_EYE_VENV:?}/bin/activate"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HOME="${THIRD_EYE_HF_HOME:?}" TOKENIZERS_PARALLELISM=false
export TMPDIR="/tmp/third_eye_a1_${USER}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" runs
python -m pytest tests -q -p no:cacheprovider --basetemp "$TMPDIR/tests"
python cluster/prepare_a1.py
