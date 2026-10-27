#!/bin/bash
#SBATCH --job-name=third-eye-revision-validation
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=third_eye_dgx_validation_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]]
root="$HOME/third_eye_dgx_20261026"
cd "${THIRD_EYE_SOURCE_ROOT:?}"
export PYTHONPATH="$PWD/src:$PWD/experiments" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
output="${THIRD_EYE_VALIDATION_OUTPUT:?}"
mkdir "$output"
"$root/.venv/bin/python" -c 'import json; from pathlib import Path; from third_eye.io import file_digest; from third_eye.provenance import source_inventory; r=json.loads(Path("release.json").read_text()); assert source_inventory()["source_tree_sha256"]==r["source_tree_sha256"]; assert all(file_digest(p)==h for p,h in r["files"].items())'
"$root/.venv/bin/python" -m pytest -q tests --basetemp "$output/fixtures" -p no:cacheprovider | tee "$output/tests.log"
"$root/.venv/bin/python" -m ruff check src experiments tests | tee "$output/lint.log"
"$root/.venv/bin/python" -c 'import sys; from third_eye.io import write_json; from third_eye.provenance import source_inventory; write_json(sys.argv[1],{"passed":True,**source_inventory()})' "$output/completed.json"
