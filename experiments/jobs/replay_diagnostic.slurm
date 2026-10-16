#!/bin/bash
#SBATCH --job-name=third-eye-replay-diagnostic
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --account=IRI23021
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=0
"$venv_root/bin/python" experiments/diagnose_replay.py \
    --campaign "${THIRD_EYE_REPLAY_CAMPAIGN:?}" \
    --validation "${THIRD_EYE_VALIDATION_OUTPUT:?}/completed.json" \
    --output "${THIRD_EYE_REPLAY_OUTPUT:?}" --index "$SLURM_ARRAY_TASK_ID"
