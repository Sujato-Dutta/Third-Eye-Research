#!/bin/bash
#SBATCH --job-name=third-eye-a2-v4-cuda-check
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_recovery_v4_cuda_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=0
python tools/recovery_a2_verify_v4.py
