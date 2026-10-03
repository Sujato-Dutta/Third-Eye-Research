# DGX execution runbook

All setup/data/verification/training executes through SLURM. The login node is
for transfer, scheduler inspection and submission. Repository code rejects
GPU execution without an allocation or on a hostname containing `login`.
Do not store cluster/Hugging Face credentials in source, configs or job logs.

## Provision and verify

Submit from the repository root under the account's home directory:

```bash
sinfo -o '%P %a %l %D %G'
squeue -u "$USER"
quota -s
sbatch experiments/jobs/bootstrap.slurm
# After the setup job succeeds:
sbatch experiments/jobs/prepare_data.slurm
sbatch --partition=gpu_student --gres=gpu:a100_1g.5gb:1 \
  --export=ALL,THIRD_EYE_VERIFY_NF4=1 experiments/jobs/verify_gpu.slurm
```

The guide names `gpu_student` and a 5GB MIG example. Use the live scheduler's
actual allowed resource names; the tiny verification fitting 5GB does not
establish that a research backbone fits. A larger MIG/full A100 allocation may
be required. Bootstrap defaults to PyTorch cu126; set `THIRD_EYE_TORCH_INDEX`
to the compatible official wheel index after inspecting the driver. Record the
runtime dependency snapshot. `HF_HOME` defaults to `models/hf_cache`.

Llama and Gemma need approved gated-model access. Configure the standard HF
authentication outside logs. Code runs also require the approved isolated
verifier described in [integration.md](integration.md). Verify container
availability on the allocated node before spending a code-label budget.
Set `THIRD_EYE_VERIFIER=module:factory` to use a trusted cluster-approved
implementation in both trajectory jobs and final evaluation.

## Pilot and freeze

```bash
export THIRD_EYE_CONFIG=experiments/configs/qwen3_4b_full_gpu_pilot.json
export THIRD_EYE_MANIFEST="$PWD/data/processed/v1/math/selection.json"
sbatch --partition=gpu_student --gres=<allowed_larger_gpu_resource> \
  experiments/jobs/pilot.slurm
```

Review verified-pool size, token budgets, throughput, memory and all K=3 H=2
branches. Uniformly adjust candidate size/LR/rank/steps only during pilots.
Freeze the settings that actually produced a completed pilot:

```bash
python experiments/freeze_protocol.py \
  --config experiments/configs/qwen3_4b_full_gpu_pilot.json \
  --pilot-run runs/pilot/<job_id> --depth 5 \
  --output experiments/configs/qwen3_4b_frozen.json
```

Repeat for Llama-small. Frozen configs pin the actual 40-character model
commit. A depth change affects the number of accepted updates, not measured
per-update training budgets. All five backbone pilot templates are supplied;
8B pilots themselves require a previously passed Gate 2; set
`THIRD_EYE_GATE2` to that decision artifact when using `pilot.slurm`.

## Staged core study

```bash
python experiments/plan_study.py \
  --configs experiments/configs/qwen3_4b_frozen.json \
            experiments/configs/llama3_2_3b_frozen.json \
  --output experiments/plans/study_v1 --run-root runs/study_v1

python experiments/submit_stage.py \
  --plan experiments/plans/study_v1/plan.json --stage labels \
  --gres=<allowed_larger_gpu_resource> \
  --estimated-task-hours=<measured_estimate> --indices 0 1 2 3
```

Submission is a dry run until `--submit` is supplied. It reserves estimated
GPU-hours for queued tasks, adds measured completed/failed wall time, rejects
an over-budget proposal, and limits arrays to one simultaneous task. It cannot
predict unknown runtimes; derive estimates from full pilot wall time including
generation, evaluation and all branches. Reconcile stale running statuses or
cancelled reservations against `sacct` before creating a versioned rerun plan.

The default core plan spans both families, three seeds, and three independent
exploration trajectories per seed. Start with a small task subset and compute
Gate 1 on its complete states before adding the remaining collection budget:

```bash
python experiments/analyze.py --labels <completed_label_files> \
  --output runs/gate1_initial
```

After Gate 1 supports continuation, finish `labels`, then `forecasters`, `forecast_reports`,
`analysis`, `online_core`, `final_core`, and `reports`. Submit each stage after
its predecessors complete; `--dependency <job_id>` adds `afterok`. The `stress`
stage evaluates seeded shuffled prompt order using preselected forecasters;
`reports_stress` exports its paired development comparisons.
Each task writes running/complete/failed status with exact command argv,
source-plan hash, host/job ID, elapsed time and storage preflight.

Direct, H=1 MLP, matched H=1 GRU, Dynamics, five feature ablations and a scalar
H=2 ablation share the same trajectory split. Online policies use the matching
forecaster training seed. Analysis reports future ranking for each main method.

## Transfer and final scores

After core Gate 2 passes, pilot/freeze Gemma and both 8B backbones. Generate a
new versioned plan with `--transfer-configs` pointing to those measured configs.
`transfer_labels` and `transfer_forecasts` measure held-out-family/scale ranking;
`online_transfer` and `final_transfer` compare no-update/greedy/Direct on the
primary task. These stages require a passed Gate 2. Rotate models sequentially
under the storage quota rather than caching all five backbones together.

Final evaluation loads only the sealed final manifest and an accepted final
adapter. It records item correctness, greedy pass@1, test-set counts and hashes.
Reports separate development trajectories from final benchmark CSVs and pair
final items by benchmark role. Core reports include both greedy and H=1
reference comparisons. No empirical success is inferred from test fixtures.

## Automatic continuation

`experiments/jobs/continue_study.slurm` runs a CPU controller, which submits
GPU work through SLURM. Set `THIRD_EYE_CAMPAIGN` to a JSON manifest containing
the four model/task pilot config paths, job IDs and run directories, plus
`data_root`, `prior_gpu_hours`, optional transfer pilot templates/public source
specifications, and `stress`. Set `THIRD_EYE_CAMPAIGN_OUTPUT` to a new output
directory. Export `THIRD_EYE_MODEL_SOURCES` and `THIRD_EYE_SANDBOX_CONFIG` as
needed. Submit after the validation job succeeds and all four pilots terminate.

The controller checks each pilot's completed artifact and equal math/code
protocols, freezes measured settings, and collects four predetermined T=5
trajectories for the first 20-state Gate 1 report. A failed or insufficient
gate stops advancement. After Gate 1 passes it completes collection, fits the
forecasters and ablations, and computes Gate 2. Online/final comparisons and
then family/scale transfer require Gate 2. Transfer models receive their own
measured pilots; the controller rotates only completed model caches to retain
storage headroom. Stress runs require sufficient remaining budget.

GPU spending includes pilot time, failed/completed task time and queued
reservations. Label estimates use measured state runtime with headroom;
matched online/final groups refine estimates after their first task. Controller
events and `status.json` expose failures, gate decisions, budget stops, cache
rotation and job IDs. Inspect those records before any versioned rerun.

## Quota and recovery

The configured account quota is 50GB. Study tasks check project footprint and
filesystem headroom; `quota -s` remains authoritative for other account usage.
Plans use `--prune-branches --keep-accepted 1`: complete-state labels, batches,
logs, adapter hashes and the latest accepted checkpoint survive. Intermediate
branch and older accepted adapters are removed only after commitment. Archive
final adapters and logs before rotating models:

```bash
python experiments/manage_cache.py
python experiments/manage_cache.py --evict-model Qwen/Qwen3-4B
```

Evict only after all jobs using that model have stopped. This utility operates
on the dedicated HF cache and the exact named model; it does not touch adapters.
Keep cache, environment and output sizes within the account's total quota.

Resume from the latest accepted checkpoint with the same config, manifest,
policy, pool-order seed and forecaster into a new output directory. The CLI
bounds resumed invocations by the remaining generations under T=5; use
`--generations` to request a smaller slice. Online resume validates forecaster
weight identity and stitches the prior baseline/trajectory points so a resumed
result still measures change from M_0. For example:

```bash
python experiments/run.py --mode online --policy direct \
  --config <same_frozen_config> --manifest <same_selection_manifest> \
  --forecaster <same_forecaster> --resume <run>/accepted/generation_2 \
  --output runs/resumed_trial --prune-branches --keep-accepted 1
```

A partial invocation cannot enter full-depth policy reports. Insufficient
corrections and overlong SFT targets require uniform repiloting, not fabricated
continuations or truncation. Partial states remain diagnostic and never enter
meta-training.
