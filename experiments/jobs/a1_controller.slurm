#!/bin/bash
#SBATCH --job-name=third-eye-a1-control
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=third_eye_a1_controller_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module load gcc cuda python3
source "${THIRD_EYE_VENV:?}/bin/activate"
export PATH="$PWD/cluster/bin:$PATH"
export HF_HOME="${THIRD_EYE_HF_HOME:?}" OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export THIRD_EYE_MODEL_SOURCES="${THIRD_EYE_ORIGINAL_ROOT:?}/experiments/model_sources_vista.json"
export THIRD_EYE_SANDBOX_CONFIG="$THIRD_EYE_ORIGINAL_ROOT/experiments/sandbox_vista.json"
export THIRD_EYE_VERIFIER=third_eye.evaluation.benchmarks:make_verifier
export THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1
review=()
if [[ -n "${THIRD_EYE_A1_REVIEWED_PILOTS:-}" ]]; then
    review=(--reviewed-pilots "$THIRD_EYE_A1_REVIEWED_PILOTS")
fi
python cluster/run_a1_pilots.py ${THIRD_EYE_A1_RESUME:+--resume} "${review[@]}"
