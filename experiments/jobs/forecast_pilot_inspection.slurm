#!/bin/bash
#SBATCH --job-name=third-eye-pilot-inspect
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --account=IRI23021
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
source_root="${THIRD_EYE_SOURCE_ROOT:?}"
venv_root="${THIRD_EYE_VENV:?}"
module purge
module load gcc cuda python3
python_binary=$(readlink -f "$venv_root/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$python_binary")")/lib:${LD_LIBRARY_PATH:-}"
cd "$source_root"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$source_root/src:$source_root/experiments"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$venv_root/bin/python" experiments/forecasting_pilots.py inspect --plan "${THIRD_EYE_PILOT_PLAN:?}"
