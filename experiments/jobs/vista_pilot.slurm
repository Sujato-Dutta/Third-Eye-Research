#!/bin/bash
#SBATCH --job-name=third-eye-pilot
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=third_eye_vista_pilot_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/experiments/jobs/vista_environment.sh"
nvidia-smi
extra_args=()
[[ -z "${THIRD_EYE_GATE2:-}" ]] || extra_args+=(--gate2 "$THIRD_EYE_GATE2")
python experiments/run.py --config "${THIRD_EYE_CONFIG:?}" \
    --manifest "${THIRD_EYE_MANIFEST:?}" --output "runs/pilot/${SLURM_JOB_ID}" \
    --pilot "${extra_args[@]}"

