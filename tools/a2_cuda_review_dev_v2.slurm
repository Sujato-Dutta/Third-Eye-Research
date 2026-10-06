#!/bin/bash
#SBATCH --job-name=third-eye-a2-cuda-review-v2
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_a2_cuda_dev_review_v2_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
python tools/review_a2_cuda_dev_v2.py
