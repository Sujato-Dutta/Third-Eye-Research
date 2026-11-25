#!/bin/bash
# Execute the frozen prospective overlap release inside a Slurm allocation.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} && $(hostname -s) != *login* ]] || exit 1

mode=$1
shift
base=/scratch/11617/sujato_ts/third_eye_ls6_20261116
work="$base/revision_20261122"
source_dir="$work/overlap_sources_20261124"
output="$work/fresh_overlap_20261124"
python="$base/.venv/bin/python"
cd "$source_dir"
sha256sum -c overlap_release.sha256
release=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$base/control/bound_horizon.json")
export PYTHONPATH="$work/replay_sources_20261123:$work:$base/control:$release/src:$release/experiments"
export HF_HOME="$base/hf_cache" HF_HUB_CACHE="$base/hf_cache/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export THIRD_EYE_VERIFIER_WORKERS=4 THIRD_EYE_SYMBOLIC_WORKERS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

common=(--request "$base/control/bound_positive.json"
        --confirmation "$base/positive/confirmation_review.json" --output "$output")
if [[ "$mode" == validate || "$mode" == freeze || "$mode" == review ]]; then
  export CUDA_VISIBLE_DEVICES=''
  exec "$python" revision_overlap_confirmation.py "$mode" "${common[@]}"
fi

[[ "$mode" == harvest || "$mode" == score ]] || exit 1
[[ "${SLURM_JOB_PARTITION:-}" == gpu-a100 && "${SLURM_JOB_NUM_NODES:-}" == 1 ]] || exit 1
lanes_text=$("$python" -c 'from ls6_runtime import require_compute, gpu_lanes; require_compute(); print(*gpu_lanes(), sep=chr(10))')
mapfile -t lanes <<< "$lanes_text"
[[ ${#lanes[@]} == 3 && $# -ge 1 && $# -le 3 ]] || exit 1
pids=()
for seed in "$@"; do
  lane=${#pids[@]}
  printf '%s seed %s on GPU %s\n' "$mode" "$seed" "${lanes[lane]}"
  CUDA_VISIBLE_DEVICES="${lanes[lane]}" "$python" revision_overlap_confirmation.py "$mode" +    "${common[@]}" --seed "$seed" > "$work/logs/overlap-$mode-$seed-$SLURM_JOB_ID.out" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
