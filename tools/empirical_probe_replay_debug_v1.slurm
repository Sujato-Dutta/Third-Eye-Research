#!/bin/bash
#SBATCH --job-name=third-eye-empirical-probe-debug
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_empirical_probe_debug_v1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=0
python tools/empirical_probe_replay_debug_v1.py
