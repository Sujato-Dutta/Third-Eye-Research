#!/bin/bash
#SBATCH --job-name=third-eye-a2-fast-forecast
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=third_eye_a2_fast_forecast_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python -m pytest -q tests tools/test_validation_source_identity.py tools/test_forecast_a2_prefix.py tools/test_forecast_a2_fast.py --basetemp "$TMPDIR/tests"
python tools/forecast_a2_prefix_fast_20261008.py
