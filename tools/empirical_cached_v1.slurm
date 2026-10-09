#!/bin/bash
#SBATCH --job-name=third-eye-empirical-cached
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_empirical_cached_v1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python -m pytest -q tests tools/test_validation_source_identity.py tools/test_empirical_v1.py --basetemp "$TMPDIR/empirical_tests"
python -c 'from tools.empirical_common_v1 import preparation; preparation()'
python tools/empirical_cached_v1.py
