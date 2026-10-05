#!/bin/bash
#SBATCH --job-name=third-eye-a2-labels
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=third_eye_a2_labels_%A_%a.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=0
python tools/a2_run_label.py
