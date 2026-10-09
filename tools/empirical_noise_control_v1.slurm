#!/bin/bash
#SBATCH --job-name=third-eye-empirical-control
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_empirical_control_v1_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES=""
if [[ "${1:?}" == validate ]]; then
  python -m pytest -q tests tools/test_validation_source_identity.py tools/test_empirical_v1.py tools/test_empirical_noise_v1.py tools/test_empirical_relay_v1.py --basetemp "$TMPDIR/noise_tests"
fi
python tools/empirical_noise_control_v1.py "${1:?}"
