#!/bin/bash
#SBATCH --job-name=third-eye-forecast-review
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --account=IRI23021
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
source_root="${THIRD_EYE_SOURCE_ROOT:?}"
venv_root="${THIRD_EYE_VENV:?}"
result_root="${THIRD_EYE_FORECAST_REVIEW_OUTPUT:?}"
pilot_root="${THIRD_EYE_PILOT_ROOT:?}"
module purge
module load gcc cuda python3
python_binary=$(readlink -f "$venv_root/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$source_root/src:$source_root/experiments" PYTHONDONTWRITEBYTECODE=1
cd "$source_root"
"$venv_root/bin/python" -c 'import json,os; from pathlib import Path; from third_eye.provenance import source_inventory; d=json.loads((Path(os.environ["THIRD_EYE_VALIDATION_OUTPUT"])/"completed.json").read_text()); assert d["passed"] and d["source_tree_sha256"]==source_inventory()["source_tree_sha256"]'
"$venv_root/bin/python" experiments/review_forecasting.py --pilot-plan "$pilot_root/plan.json" --inspection "$pilot_root/pilot_inspection.json" --output "$result_root" --frozen-forecaster "${THIRD_EYE_FROZEN_FORECASTER:?}" --frozen-weights-sha256 b0e4ec855be88adc6a98f224df70ce12cf3c79d16af786074cd6c8607d68c2e4
