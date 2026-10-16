#!/bin/bash
#SBATCH --job-name=third-eye-repeat-cuda
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=IRI23021
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
source_root="${THIRD_EYE_SOURCE_ROOT:?}"
venv_root="${THIRD_EYE_VENV:?}"
result_root="${THIRD_EYE_CUDA_VALIDATION_OUTPUT:?}"
module purge
module load gcc cuda python3
python_binary=$(readlink -f "$venv_root/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$python_binary")")/lib:${LD_LIBRARY_PATH:-}"
mkdir "$result_root"
cd "$source_root"
export CUDA_VISIBLE_DEVICES=0 THIRD_EYE_REPEAT_VERIFY_DEVICE=cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$source_root/src:$source_root/experiments"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$venv_root/bin/python" experiments/verify_backend.py --device cuda --output "$result_root/backend.json"
"$venv_root/bin/python" -m pytest -c "$source_root/pytest.ini" --rootdir "$source_root" -q tests/test_repeat_collection.py --basetemp "$result_root/fixtures" -p no:cacheprovider | tee "$result_root/tests.log"
export THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1
"$venv_root/bin/python" experiments/forecasting_pilots.py preflight --plan "${THIRD_EYE_PILOT_PLAN:?}"
"$venv_root/bin/python" -c 'from third_eye.provenance import source_inventory; from third_eye.io import write_json; import sys; write_json(sys.argv[1], {"passed":True, **source_inventory()})' "$result_root/completed.json"
