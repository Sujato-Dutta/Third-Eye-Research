#!/bin/bash
#SBATCH --job-name=third-eye-final-parents
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-7%5
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=0
"$venv_root/bin/python" experiments/finish_confirmation.py run --request "${THIRD_EYE_CONFIRMATION_REQUEST:?}" --index "$((SLURM_ARRAY_TASK_ID + ${THIRD_EYE_INDEX_BASE:?}))"
