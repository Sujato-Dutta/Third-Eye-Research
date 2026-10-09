#!/bin/bash
#SBATCH --job-name=third-eye-a2-v4-handoff-check
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_recovery_v4_handoff_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python tools/recovery_a2_handoff_verify_v4.py
