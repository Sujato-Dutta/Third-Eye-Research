#!/bin/bash
#SBATCH --job-name=third-eye-e2-cuda-check
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=third_eye_e2_cuda_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=0
export THIRD_EYE_EMPIRICAL_QUEUE="$SLURM_SUBMIT_DIR/runs/a2/empirical_v1/checkpoint_fallback_v3/submission_queue"
python tools/empirical_checkpoint_v3.py verify
