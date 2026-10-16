#!/bin/bash
#SBATCH --job-name=third-eye-crossed-review
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=""
"$venv_root/bin/python" experiments/crossed_corrections.py inspect --plan "${THIRD_EYE_CROSSED_PLAN:?}"
