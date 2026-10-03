#!/bin/bash
#SBATCH --job-name=third-eye-pilot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=third_eye_pilot_%j.out
# Supply the account's allowed GPU resource/partition when submitting.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export OMP_NUM_THREADS="${THIRD_EYE_OMP_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
extra_args=()
if [[ -n "${THIRD_EYE_GATE2:-}" ]]; then
    extra_args+=(--gate2 "$THIRD_EYE_GATE2")
fi
python experiments/run.py \
    --config "${THIRD_EYE_CONFIG:-experiments/configs/qwen3_4b_pilot.json}" \
    --manifest "${THIRD_EYE_MANIFEST:?Set the immutable development split manifest}" \
    --output "runs/pilot/${SLURM_JOB_ID}" --pilot "${extra_args[@]}"
