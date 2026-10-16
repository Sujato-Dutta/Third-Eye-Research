#!/bin/bash
#SBATCH --job-name=third-eye-crossed-corrections
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-23%20
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=0
"$venv_root/bin/python" experiments/crossed_corrections.py run --plan "${THIRD_EYE_CROSSED_PLAN:?}" --index "$SLURM_ARRAY_TASK_ID"
