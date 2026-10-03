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
python experiments/launch_vista.py --preflight-job "${THIRD_EYE_PREFLIGHT_JOB:?}" \
    --account "${THIRD_EYE_ACCOUNT:?}" --max-parallel "${THIRD_EYE_MAX_PARALLEL:?}" \
    --budget-hours "${THIRD_EYE_BUDGET_HOURS:?}" \
    --campaign "${THIRD_EYE_CAMPAIGN:?}" --output "${THIRD_EYE_CAMPAIGN_OUTPUT:?}"
