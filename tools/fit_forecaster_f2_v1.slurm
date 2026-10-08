#!/bin/bash
#SBATCH --job-name=third-eye-forecaster-f2
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=third_eye_forecaster_f2_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python -m pytest -q tests tools/test_validation_source_identity.py tools/test_forecaster_f2.py --basetemp "$TMPDIR/tests"
python tools/fit_forecaster_f2_v1.py
