#!/bin/bash
#SBATCH --job-name=third-eye-a2-recovery
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=third_eye_a2_recovery_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=0
python tools/recovery_a2_controller_v2.py resume --index "${1:?}"
