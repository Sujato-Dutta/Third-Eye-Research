#!/bin/bash
#SBATCH --job-name=third-eye-a2-recovery-cpu
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_a2_recovery_cpu_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
if [[ "${1:?}" == prepare ]]; then
    python -m pytest -q tests tools/test_validation_source_identity.py tools/test_recovery_a2.py tools/test_recovery_submission_v2.py --basetemp "$TMPDIR/tests"
fi
python tools/recovery_a2_controller_v2.py "$@"
