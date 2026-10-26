#!/bin/bash
#SBATCH --job-name=third-eye-smol16-parents
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --array=0-7%8
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=0
"$venv_root/bin/python" experiments/smollm_confirmation.py run --request "${THIRD_EYE_SMOL_REQUEST:?}" --index "$((SLURM_ARRAY_TASK_ID + ${THIRD_EYE_INDEX_BASE:?}))"
