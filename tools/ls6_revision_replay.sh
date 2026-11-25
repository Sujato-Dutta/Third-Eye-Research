#!/bin/bash
# Run only in a Lonestar6 Slurm allocation. The launcher binds one frozen release.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} && $(hostname -s) != *login* ]] || exit 1

mode=$1
group=$2
shift 2
base=/scratch/11617/sujato_ts/third_eye_ls6_20261116
work="$base/revision_20261122"
source_dir="$work/replay_sources_20261123"
python="$base/.venv/bin/python"
request="$base/control/bound_positive.json"
horizon="$base/control/bound_horizon.json"

cd "$source_dir"
sha256sum -c replay_release.sha256
release=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$horizon")
export PYTHONPATH="$work:$base/control:$release/src:$release/experiments"
export HF_HOME="$base/hf_cache" HF_HUB_CACHE="$base/hf_cache/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export THIRD_EYE_VERIFIER_WORKERS=4 THIRD_EYE_SYMBOLIC_WORKERS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
common=(--work "$work" --request "$request" --registry "$work/qwen4_registry.json"
        --confirmation "$base/positive/confirmation_review.json")
mkdir -p "$work"/{replay,pricing,replay_validation}
receipt="$work/replay_validation/$group.json"

if [[ "$mode" == validate ]]; then
  export CUDA_VISIBLE_DEVICES=''
  "$python" revision_replay.py validate "${common[@]}" --seeds "$@" --output "$receipt"
  exit
fi

if [[ "$mode" == review ]]; then
  export CUDA_VISIBLE_DEVICES=''
  "$python" revision_replay.py review "${common[@]}" --outcomes \
    "$work/dgx_saved_outcomes.npz" "$work/lonestar_saved_outcomes.npz" \
    --output "$work/archived_probe_review.json"
  exit
fi

[[ "$mode" == replay || "$mode" == price ]] || exit 1
[[ "${SLURM_JOB_PARTITION:-}" == gpu-a100-dev || "${SLURM_JOB_PARTITION:-}" == gpu-a100 ]] || exit 1
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && -s "$receipt" ]] || exit 1
lanes_text=$("$python" -c 'from ls6_runtime import require_compute, gpu_lanes; require_compute(); print(*gpu_lanes(), sep=chr(10))')
mapfile -t lanes <<< "$lanes_text"
[[ ${#lanes[@]} == 3 ]] || exit 1

seeds=("$@")
out="$work/replay"
if [[ "$mode" == price ]]; then
  out="$work/pricing"
fi
worker() {
  local lane=$1
  local index seed dest
  local status=0
  for ((index=lane; index<${#seeds[@]}; index+=${#lanes[@]})); do
    seed=${seeds[index]}
    dest="$out/$seed.json"
    if [[ -s "$dest" ]]; then
      printf 'Preserved existing output for %s\n' "$seed"
      continue
    fi
    printf 'Starting %s seed %s on GPU %s\n' "$mode" "$seed" "${lanes[lane]}"
    CUDA_VISIBLE_DEVICES="${lanes[lane]}" "$python" revision_replay.py "$mode" \
      "${common[@]}" --validated "$receipt" --seed "$seed" --output "$dest" \
      > "$work/logs/$mode-$seed-$SLURM_JOB_ID.out" 2>&1 || status=1
  done
  return "$status"
}

pids=()
for ((lane=0; lane<${#lanes[@]} && lane<${#seeds[@]}; lane++)); do
  worker "$lane" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
