#!/bin/bash
#SBATCH --job-name=third-eye-empirical-comparator
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_empirical_comparator_v1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
if [[ "${1:?}" == fit ]]; then
  third_eye_comparator_env="$SLURM_SUBMIT_DIR/runs/a2/empirical_v1/comparator_venv"
  [[ ! -e "$third_eye_comparator_env" ]] || exit 1
  python -m venv --system-site-packages "$third_eye_comparator_env"
  "$third_eye_comparator_env/bin/python" -m pip install --no-deps --only-binary=:all: lightgbm==4.6.0 scikit-learn==1.7.2 scipy==1.16.2 joblib==1.5.2 threadpoolctl==3.6.0
  "$third_eye_comparator_env/bin/python" -m pytest -q tools/test_empirical_comparator_v1.py --basetemp "$TMPDIR/comparator_tests"
  "$third_eye_comparator_env/bin/python" tools/empirical_comparator_v1.py fit
else
  python tools/empirical_comparator_v1.py "${1:?}"
fi
