#!/bin/bash
#SBATCH --job-name=third-eye-controller
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=third_eye_vista_controller_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module load gcc cuda python3
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export THIRD_EYE_SANDBOX_CONFIG="${THIRD_EYE_SANDBOX_CONFIG:-$PWD/experiments/sandbox_vista.json}"
python experiments/continue_study.py --cluster vista \
    --account "${THIRD_EYE_ACCOUNT:?Verified Vista allocation required}" \
    --max-parallel "${THIRD_EYE_MAX_PARALLEL:?Verified queue/account limit required}" \
    --budget-hours "${THIRD_EYE_BUDGET_HOURS:?Measured allocation budget required}" \
    --storage-quota-gb "${THIRD_EYE_STORAGE_QUOTA_GB:-50}" \
    --campaign "${THIRD_EYE_CAMPAIGN:?}" --output "${THIRD_EYE_CAMPAIGN_OUTPUT:?}"

