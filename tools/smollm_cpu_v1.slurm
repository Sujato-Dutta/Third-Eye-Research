#!/bin/bash
#SBATCH --job-name=third-eye-smollm-cpu
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=third_eye_smollm_cpu_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
if [[ "${1:?}" == setup ]]; then
    python -m pytest -q tests tools/test_validation_source_identity.py tools/test_smollm_supplement_v1.py --basetemp "$TMPDIR/smollm_tests"
fi
python tools/smollm_supplement_v1.py "$@"
