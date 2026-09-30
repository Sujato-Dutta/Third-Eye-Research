#!/bin/bash
#SBATCH --job-name=third-eye-verify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=third_eye_verify_%j.out
# Supply --partition and --gres at submission to match your account/allocation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
python experiments/verify_backend.py --device cuda --output "runs/gpu_verification/${SLURM_JOB_ID}/lora.json"
if [[ "${THIRD_EYE_VERIFY_NF4:-0}" == "1" ]]; then
    python experiments/verify_backend.py --device cuda --quantization nf4 --output "runs/gpu_verification/${SLURM_JOB_ID}/qlora.json"
fi
python -m pytest -q
