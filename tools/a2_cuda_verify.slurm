#!/bin/bash
#SBATCH --job-name=third-eye-a2-cuda-verify
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=third_eye_a2_cuda_%A_%a.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=0
python tools/verify_a2_cuda.py
export THIRD_EYE_A1_VERIFY_DEVICE=cuda
python -m pytest -q tests/test_a1.py -k terminal_continuation \
  -p no:cacheprovider --basetemp "$TMPDIR/terminal_tests"
