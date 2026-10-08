#!/bin/bash
#SBATCH --job-name=third-eye-signal-v1
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:05:00
#SBATCH --output=third_eye_signal_v1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python -m pytest -q tests tools/test_validation_source_identity.py tools/test_forecaster_f2.py tools/test_diagnose_forecast_signal_v1.py --basetemp "$TMPDIR/tests"
python tools/diagnose_forecast_signal_v1.py
