#!/bin/bash
#SBATCH --job-name=third-eye-a2-cpu-forecasters
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=third_eye_a2_forecasters_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
python tools/a2_after_labels.py
