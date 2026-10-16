#!/bin/bash
#SBATCH --job-name=third-eye-validation
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --account=IRI23021

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == *login* ]]; then
    echo 'Validation requires an allocated compute node' >&2
    exit 1
fi
source_root="${THIRD_EYE_SOURCE_ROOT:?Set an immutable snapshot of the current main implementation}"
venv_root="${THIRD_EYE_VENV:?Set the shared project virtual environment}"
result_root="${THIRD_EYE_VALIDATION_OUTPUT:?Set a new validation output directory}"
module purge
module load gcc cuda python3
python_binary=$(readlink -f "$venv_root/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$python_binary")")/lib:${LD_LIBRARY_PATH:-}"
mkdir "$result_root"
cd "$source_root"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$source_root/src:$source_root/experiments"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
"$venv_root/bin/python" -c 'import json,sys; from pathlib import Path; from third_eye.io import file_digest; from third_eye.provenance import source_inventory; r=json.loads(Path("release.json").read_text()); assert r["source_tree_sha256"]==source_inventory()["source_tree_sha256"]; assert all(file_digest(p)==h for p,h in r["files"].items())'
"$venv_root/bin/python" -m pytest -c "$source_root/pytest.ini" --rootdir "$source_root" -q tests --basetemp "$result_root/fixtures" -p no:cacheprovider | tee "$result_root/tests.log"
"$venv_root/bin/python" -m ruff check src experiments tests | tee "$result_root/lint.log"
"$venv_root/bin/python" -c 'from third_eye.provenance import source_inventory; from third_eye.io import write_json; import sys; write_json(sys.argv[1], {"passed":True, **source_inventory()})' "$result_root/completed.json"
