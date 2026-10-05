#!/bin/bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load gcc cuda python3
source "${THIRD_EYE_VENV:?}/bin/activate"
third_eye_python_binary=$(readlink -f "$THIRD_EYE_VENV/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$third_eye_python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${THIRD_EYE_HF_HOME:?}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export THIRD_EYE_MODEL_SOURCES="${THIRD_EYE_ORIGINAL_ROOT:?}/experiments/model_sources_vista.json"
export THIRD_EYE_SANDBOX_CONFIG="$THIRD_EYE_ORIGINAL_ROOT/experiments/sandbox_vista.json"
export THIRD_EYE_VERIFIER=third_eye.evaluation.benchmarks:make_verifier
export THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="/tmp/third_eye_a2_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
