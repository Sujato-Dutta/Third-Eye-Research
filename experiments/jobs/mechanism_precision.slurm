#!/bin/bash
#SBATCH --job-name=third-eye-mechanism-precision
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=""
"$venv_root/bin/python" -c 'import json,os; from pathlib import Path; from third_eye.provenance import source_inventory; d=json.loads((Path(os.environ["THIRD_EYE_VALIDATION_OUTPUT"])/"completed.json").read_text()); assert d["passed"] and d["source_tree_sha256"]==source_inventory()["source_tree_sha256"]'
"$venv_root/bin/python" experiments/review_mechanism_precision.py --campaign "${THIRD_EYE_MECHANISM_CAMPAIGN:?}" --report-sha256 "${THIRD_EYE_MECHANISM_REPORT_SHA:?}" --output "${THIRD_EYE_PRECISION_OUTPUT:?}"
