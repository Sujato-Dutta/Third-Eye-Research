# Running and recovering training jobs

Run from the repository root. Install dependencies in `.venv`; SLURM scripts
honor an alternative path through `THIRD_EYE_VENV`. Never train on a cluster
login node. Submit work through the scheduler.

## GPU verification on MU DGX

The supplied DGX guide specifies campus-network access and the `gpu_student`
partition. Resource names and entitlements depend on the current cluster
configuration. Check `sinfo` and your account limits before submitting.

```bash
sbatch --partition=gpu_student --gres=gpu:a100_1g.5gb:1 \
  experiments/jobs/verify_gpu.slurm
```

This tiny-model smoke job fits a small MIG allocation. To additionally test
NF4, install `requirements-gpu.txt` and submit with
`--export=ALL,THIRD_EYE_VERIFY_NF4=1`. Inspect the job's output and the JSON
reports under `runs/gpu_verification/<job_id>/`.

## Qwen/Llama pilot

A tiny-model smoke job does not establish that a 4B model fits a 5GB MIG.
Choose a full or larger MIG resource appropriate to your model/config and
account. Validate the actual allocation with `nvidia-smi` inside the job.

```bash
export THIRD_EYE_MANIFEST="$PWD/data/manifests/math_v1.json"
export THIRD_EYE_CONFIG=experiments/configs/qwen3_4b_pilot.json
sbatch --partition=<your_allowed_partition> \
  --gres=<your_allowed_gpu_resource> experiments/jobs/pilot.slurm
```

For Llama-small, set `THIRD_EYE_CONFIG` to
`experiments/configs/llama3_2_3b_pilot.json` after Qwen labeling is stable.
Authenticate to HF outside logs if gated weights require it. Never commit
tokens, SSH keys, scheduler credentials, or the GPU access document.

## Resume completed accepted updates

```bash
python experiments/run.py \
  --config experiments/configs/qwen3_4b_frozen.json \
  --manifest data/manifests/math_v1.json \
  --resume runs/trajectory_part1/accepted/generation_1 \
  --output runs/trajectory_part2 --policy random
```

The config, resolved model revision, split manifest, and policy must match.
With depth=1 this adds one accepted generation. The runner restores cumulative
adapter weights, accepted history, and generation count; it resets AdamW for
the next update. It does not resume halfway through a training step. Keep the
same manifest paths on that machine or materialize a documented relocated
manifest as a distinct experiment version.

## Failures and diagnostics

- Insufficient verified corrections: review `correction_pool.json` and
  `failure.json`. Repilot a smaller **uniform** candidate size or enlarge the
  fixed training pool. Do not shrink one candidate or invent its H=2 label.
- Sequence too long: change the pilot token budget or pre-audit the data.
  Assistant targets are never silently truncated. The revised budget must be
  shared across candidates and recorded in a new config version.
- OOM: lower micro-batch size and increase accumulation consistently, enable
  NF4/checkpointing, or request a larger allocation. Repilot/freeze again.
- Partial state: completed branch checkpoints stay for diagnosis. Only a
  complete matched K=3 state is published as labels. Restart from the last
  accepted checkpoint into a new output directory; old partial files remain
  available for audit.
- Config/checkpoint mismatch: use the exact saved protocol and pinned base
  revision. Weight checksums and adapter hashes reject incompatible/corrupt
  checkpoints.

The ledger profiles state wall time, individual update time, probe time,
optimizer steps, trainable parameters, and peak allocated GPU memory. CUDA
synchronization surrounds measured update time. These measurements support
pilot budgeting; end-to-end policy cost still includes generation/evaluation,
and GPU-specific performance must be measured on the actual allocation.
