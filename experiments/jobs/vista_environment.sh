#!/bin/bash
# Source only inside a single-node Grace Hopper allocation.
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || {
    echo 'Vista execution requires a SLURM compute-node allocation' >&2; exit 1;
}
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load gcc cuda python3
module list
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
export OMP_NUM_THREADS="${THIRD_EYE_OMP_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export THIRD_EYE_VERIFIER=third_eye.evaluation.benchmarks:make_verifier
export THIRD_EYE_VERIFIER_WORKERS="${THIRD_EYE_VERIFIER_WORKERS:-16}"
export THIRD_EYE_SYMBOLIC_WORKERS=1
export TMPDIR="${SLURM_TMPDIR:-/tmp/third_eye_${USER}_${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
# Vista assigns one complete GPU node without a --gres directive. Confirm
# CUDA availability below; never infer an allocation on a login node.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
python -c 'import torch; assert torch.cuda.is_available(), "Allocated Vista node has no working CUDA runtime"'
export THIRD_EYE_SANDBOX_CONFIG="${THIRD_EYE_SANDBOX_CONFIG:-$PWD/experiments/sandbox_vista.json}"

