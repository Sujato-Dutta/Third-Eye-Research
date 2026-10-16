#!/bin/bash
#SBATCH --job-name=third-eye-continuation-review
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --account=IRI23021

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *login* ]] || exit 1
source_root="${THIRD_EYE_SOURCE_ROOT:?}"
venv_root="${THIRD_EYE_VENV:?}"
validation_root="${THIRD_EYE_VALIDATION_OUTPUT:?}"
result_root="${THIRD_EYE_CONTINUATION_OUTPUT:?}"
control_root="${THIRD_EYE_CONTROL_ROOT:?}"
module purge
module load gcc cuda python3
python_binary=$(readlink -f "$venv_root/bin/python")
export LD_LIBRARY_PATH="$(dirname "$(dirname "$python_binary")")/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$source_root/src:$source_root/experiments"
export PYTHONDONTWRITEBYTECODE=1
cd "$source_root"
"$venv_root/bin/python" -c 'import json,sys; from third_eye.provenance import source_inventory; p=json.load(open(sys.argv[1])); assert p["passed"] and p["source_tree_sha256"]==source_inventory()["source_tree_sha256"]' "$validation_root/completed.json"
native_inputs=()
controls=()
for index in {0..7}; do
    native_inputs+=("$control_root/noise_inputs/task_$index.json")
    controls+=("$control_root/noise/task_$index/completed.json")
done
"$venv_root/bin/python" experiments/review_continuations.py \
    --native-inputs "${native_inputs[@]}" --controls "${controls[@]}" \
    --output "$result_root" --exploratory-fit
