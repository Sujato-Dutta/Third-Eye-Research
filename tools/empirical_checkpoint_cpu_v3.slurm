#!/bin/bash
#SBATCH --job-name=third-eye-e2-cpu
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_e2_cpu_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
third_eye_e2_python="$SLURM_SUBMIT_DIR/runs/a2/empirical_v1/comparator_venv/bin/python"
if [[ "${1:?}" == validate ]]; then
    python -m pytest -q tests tools/test_validation_source_identity.py tools/test_empirical_v1.py tools/test_empirical_noise_v1.py tools/test_empirical_relay_v1.py tools/test_empirical_comparator_v1.py tools/test_empirical_replication_v1.py tools/test_empirical_checkpoint_v3.py tools/test_empirical_checkpoint_prequeue_v2.py --basetemp "$TMPDIR/e2_tests"
fi
if [[ "${1:?}" == confirm ]]; then
    "$third_eye_e2_python" tools/empirical_checkpoint_v3.py "$@"
else
    python tools/empirical_checkpoint_v3.py "$@"
fi
