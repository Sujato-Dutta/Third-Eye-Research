#!/bin/bash
#SBATCH --job-name=third-eye-fixture-classification
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=""
"$venv_root/bin/python" experiments/classify_scoring_fixtures.py --plan "${THIRD_EYE_FIXTURE_PLAN:?}"
