#!/bin/bash
#SBATCH --job-name=third-eye-a2-cpu-v2
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_a2_cpu_v2_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
python -m pytest -q tests tools/test_a1_parallel_resume.py tools/test_retry_budget.py \
  tools/test_validation_source_identity.py -p no:cacheprovider --basetemp "$TMPDIR/tests"
python -m py_compile tools/verify_a2_cuda_v2.py tools/review_a2_cuda_dev_v2.py tools/validate_a2_v2_cpu.py
python tools/validate_a2_v2_cpu.py
