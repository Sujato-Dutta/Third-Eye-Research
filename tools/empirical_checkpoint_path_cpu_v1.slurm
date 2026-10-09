#!/bin/bash
#SBATCH --job-name=third-eye-e2-path-check
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_e2_path_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
if [[ "${1:?}" == validate ]]; then
    python -m pytest -q tests tools/test_validation_source_identity.py tools/test_empirical_v1.py tools/test_empirical_noise_v1.py tools/test_empirical_relay_v1.py tools/test_empirical_comparator_v1.py tools/test_empirical_replication_v1.py tools/test_empirical_checkpoint_v3.py tools/test_empirical_checkpoint_prequeue_v2.py tools/test_empirical_checkpoint_path_repair_v1.py --basetemp "$TMPDIR/path_tests"
    python tools/empirical_checkpoint_path_repair_v1.py validate
else
    "$SLURM_SUBMIT_DIR/runs/a2/empirical_v1/comparator_venv/bin/python" tools/empirical_checkpoint_path_repair_v1.py "$@"
fi
