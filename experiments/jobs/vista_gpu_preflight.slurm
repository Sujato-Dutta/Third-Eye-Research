#!/bin/bash
#SBATCH --job-name=third-eye-preflight
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=third_eye_vista_preflight_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/experiments/jobs/vista_environment.sh"
export THIRD_EYE_MODEL_SOURCES="$PWD/experiments/model_sources_vista.json"
nvidia-smi
python experiments/verify_backend.py --device cuda --quantization none \
    --output "runs/vista_backend_${SLURM_JOB_ID}.json"
python experiments/profile_gpu.py --quantization none \
    --config experiments/configs/qwen3_4b_vista_pilot.json \
    --output "runs/vista_gpu_profile_${SLURM_JOB_ID}.json"
python experiments/verify_vista_preflight.py --cpu-job "${THIRD_EYE_CPU_SETUP_JOB:?}"

