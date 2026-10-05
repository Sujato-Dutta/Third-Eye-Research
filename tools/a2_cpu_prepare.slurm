#!/bin/bash
#SBATCH --job-name=third-eye-a2-cpu-validate
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_a2_cpu_%j.out
source "${SLURM_SUBMIT_DIR:?}/tools/a2_environment.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
python -m pytest -q tests tools/test_a1_parallel_resume.py tools/test_retry_budget.py \
  -p no:cacheprovider --basetemp "$TMPDIR/tests"
python -m py_compile tools/prepare_a2_study.py tools/verify_a2_cuda.py tools/a2_after_labels.py
python tools/prepare_a2_study.py
