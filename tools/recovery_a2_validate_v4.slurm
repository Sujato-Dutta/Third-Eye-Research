#!/bin/bash
#SBATCH --job-name=third-eye-a2-v4-cpu-check
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_recovery_v4_cpu_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
python -m pytest -q tests tools/test_validation_source_identity.py tools/test_recovery_a2.py tools/test_recovery_submission_v2.py tools/test_recovery_relay_v3.py tools/test_recovery_relay_v4.py tools/test_recovery_v4.py --basetemp "$TMPDIR/recovery_v4_tests"
