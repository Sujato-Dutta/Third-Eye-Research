#!/bin/bash
#SBATCH --job-name=third-eye-empirical-gpucheck
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=third_eye_empirical_gpucheck_v1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=0
python tools/empirical_noise_v1.py verify
