#!/bin/bash
#SBATCH --job-name=third-eye-setup
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=third_eye_vista_setup_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
hostname
printf '%s\n' "$SLURM_JOB_ID"
if [[ -x /usr/local/etc/taccinfo ]]; then
    /usr/local/etc/taccinfo
else
    cat vista_allocation_inventory.json
fi
module load gcc cuda python3
module list
uname -m
nvidia-smi
export TMPDIR="${SLURM_TMPDIR:-/tmp/third_eye_${USER}_${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR" runs
export PIP_NO_CACHE_DIR=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="${THIRD_EYE_HF_HOME:-$PWD/models/hf_cache}"
python3 -m venv "${THIRD_EYE_VENV:-.venv}"
source "${THIRD_EYE_VENV:-.venv}/bin/activate"
# Keep torch/model packages pinned; Vista needs CUDA wheels for Linux aarch64.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-evaluation.txt
python -m pip freeze > "runs/vista_dependencies_${SLURM_JOB_ID}.txt"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_properties(0))'
python -m pytest tests -q -p no:cacheprovider --basetemp "$TMPDIR/tests"
python experiments/verify_backend.py --device cuda --quantization none \
    --output "runs/vista_backend_${SLURM_JOB_ID}.json"
python experiments/prepare_vista.py
export THIRD_EYE_MODEL_SOURCES="$PWD/experiments/model_sources_vista.json"
python experiments/profile_gpu.py --quantization none \
    --config experiments/configs/qwen3_4b_vista_pilot.json \
    --output "runs/vista_gpu_profile_${SLURM_JOB_ID}.json"
