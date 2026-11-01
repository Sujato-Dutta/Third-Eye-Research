#!/bin/bash
#SBATCH --job-name=third-eye-llama8-cpu
#SBATCH --partition=gg
#SBATCH --account=IRI23021
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
source "${THIRD_EYE_SOURCE_ROOT:?}/experiments/jobs/bounded_evidence_environment.sh"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1
"$venv_root/bin/python" experiments/scale_confirmation.py "${THIRD_EYE_SCALE_ACTION:?}" --request "${THIRD_EYE_SCALE_REQUEST:?}" --model llama3.1-8b
