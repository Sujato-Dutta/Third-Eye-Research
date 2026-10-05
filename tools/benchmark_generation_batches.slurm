#!/bin/bash
#SBATCH --job-name=third-eye-batch-benchmark
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=third_eye_batch_benchmark_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
module purge
module load gcc cuda python3
source "${THIRD_EYE_VENV:?}/bin/activate"
third_eye_python_binary=$(readlink -f "$THIRD_EYE_VENV/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$third_eye_python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export THIRD_EYE_MODEL_SOURCES="${THIRD_EYE_ORIGINAL_ROOT:?}/experiments/model_sources_vista.json"
export THIRD_EYE_SANDBOX_CONFIG="$THIRD_EYE_ORIGINAL_ROOT/experiments/sandbox_vista.json"
export HF_HOME="${THIRD_EYE_HF_HOME:?}" CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
for model in qwen3_4b llama3_2_3b; do
    python tools/benchmark_generation_batches.py \
      --config "runs/campaign_a1_20261003/configs/${model}_1042.json" \
      --model-key "$model" --data-root data/processed/a1 \
      --pilot-root runs/campaign_a1_20261003/pilots \
      --output "runs/profiling/${SLURM_JOB_ID}/${model}_batching.json"
done
