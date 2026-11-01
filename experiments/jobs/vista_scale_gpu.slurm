#!/bin/bash
#SBATCH --job-name=third-eye-llama8-gpu
#SBATCH --partition=gh
#SBATCH --account=IRI23021
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=0
export THIRD_EYE_MODEL_SOURCES="${THIRD_EYE_SCALE_OUTPUT:?}/llama3.1-8b/model_sources.json"
arguments=("${THIRD_EYE_SCALE_ACTION:?}" --request "${THIRD_EYE_SCALE_REQUEST:?}" --model llama3.1-8b)
if [[ "$THIRD_EYE_SCALE_ACTION" == run ]]; then arguments+=(--index "${SLURM_ARRAY_TASK_ID:?}"); fi
"$venv_root/bin/python" experiments/scale_confirmation.py "${arguments[@]}"
