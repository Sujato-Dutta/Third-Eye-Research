#!/bin/bash
#SBATCH --job-name=third-eye-scoring-timing
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=0
"$venv_root/bin/python" experiments/scoring_precision.py benchmark --plan "${THIRD_EYE_SCORING_PLAN:?}" --index "$SLURM_ARRAY_TASK_ID"
