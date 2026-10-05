#!/bin/bash
#SBATCH --job-name=third-eye-a1-parallel-control
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=third_eye_a1_parallel_controller_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module purge
module load gcc cuda python3
source "${THIRD_EYE_VENV:?}/bin/activate"
third_eye_python_binary=$(readlink -f "$THIRD_EYE_VENV/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$third_eye_python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export PATH="$PWD/cluster/bin:$PATH"
export HF_HOME="${THIRD_EYE_HF_HOME:?}" OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export THIRD_EYE_MODEL_SOURCES="${THIRD_EYE_ORIGINAL_ROOT:?}/experiments/model_sources_vista.json"
export THIRD_EYE_SANDBOX_CONFIG="$THIRD_EYE_ORIGINAL_ROOT/experiments/sandbox_vista.json"
export THIRD_EYE_VERIFIER=third_eye.evaluation.benchmarks:make_verifier
export THIRD_EYE_VERIFIER_WORKERS=16 THIRD_EYE_SYMBOLIC_WORKERS=1
python tools/a1_parallel_resume.py --old-controller "${THIRD_EYE_OLD_CONTROLLER:?}" \
  --max-parallel 20 --budget-hours 3000 \
  --allocation-available "${THIRD_EYE_ALLOCATION_AVAILABLE:?}"
