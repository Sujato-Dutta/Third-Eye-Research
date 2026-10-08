#!/bin/bash
#SBATCH --job-name=third-eye-forecaster-diagnostic
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_a2_forecaster_diagnostic_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python -m pytest -q tools/test_forecast_a2_patience_diag.py
python tools/forecast_a2_patience_diag_20261008.py
