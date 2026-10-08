#!/bin/bash
#SBATCH --job-name=third-eye-signal-confirm
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:05:00
#SBATCH --output=third_eye_signal_confirm_v1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python -m pytest -q tools/test_confirm_forecast_signal_v1.py --basetemp "$TMPDIR/confirmation_tests"
python tools/confirm_forecast_signal_v1.py
