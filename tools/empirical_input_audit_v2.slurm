#!/bin/bash
#SBATCH --job-name=third-eye-empirical-input-audit
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_empirical_input_audit_v2_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python tools/empirical_input_audit_v2.py
