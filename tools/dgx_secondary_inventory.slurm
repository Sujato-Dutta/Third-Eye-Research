#!/bin/bash
#SBATCH --job-name=te-account-inventory
#SBATCH --account=scholar
#SBATCH --partition=cpu_scholar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --chdir=/dgxb_home/se23pcse004
#SBATCH --output=/dgxb_home/se23pcse004/third_eye_account_audit_20261109/logs/inventory-%j.out
#SBATCH --error=/dgxb_home/se23pcse004/third_eye_account_audit_20261109/logs/inventory-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 /dgxb_home/se23pcse004/third_eye_account_audit_20261109/storage.py snapshot --output /dgxb_home/se23pcse004/third_eye_account_audit_20261109
