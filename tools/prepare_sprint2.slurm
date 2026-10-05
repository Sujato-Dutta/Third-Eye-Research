#!/bin/bash
#SBATCH --job-name=third-eye-sprint2-prepare
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_sprint2_prepare_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module purge
module load gcc cuda python3
source "${THIRD_EYE_VENV:?}/bin/activate"
third_eye_python_binary=$(readlink -f "$THIRD_EYE_VENV/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$third_eye_python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TMPDIR=/tmp/third_eye_sprint2_prepare
mkdir -p "$TMPDIR"
python -m pytest -q tests tools/test_a1_parallel_resume.py tools/test_retry_budget.py \
  -p no:cacheprovider --basetemp "$TMPDIR/tests"
python -m py_compile tools/paired_retry_control.py tools/prepare_sprint2.py
python tools/prepare_sprint2.py
