#!/bin/bash
#SBATCH --job-name=third-eye-empirical-methods
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_empirical_methods_v1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
third_eye_replication_python="$SLURM_SUBMIT_DIR/runs/a2/empirical_v1/comparator_venv/bin/python"
"$third_eye_replication_python" -m pytest -q tools/test_empirical_replication_v1.py --basetemp "$TMPDIR/replication_tests"
"$third_eye_replication_python" tools/empirical_replication_v1.py "${1:?}"
