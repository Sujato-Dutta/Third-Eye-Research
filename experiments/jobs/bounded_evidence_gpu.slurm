#!/bin/bash
#SBATCH --job-name=third-eye-half-lr
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=0
"$venv_root/bin/python" experiments/strengthen_evidence.py run --plan "${THIRD_EYE_EVIDENCE_PLAN:?}" --index "$SLURM_ARRAY_TASK_ID"
