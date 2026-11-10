# Paired longer-horizon sensitivity

Frozen November 9, 2026, before new trajectory outcomes. The existing twenty-parent
two-update confirmation retains its complete seed grid, settings and inference.

## Scope

Four paired seeds on each of four backbones: Qwen3-4B, Llama-3.2-3B-Instruct,
Qwen3-8B and Llama-3.1-8B-Instruct. These span **two families and two scales**,
not four independent families. Math is the target task; GSM8K, MATH-500 and MMLU
measure target, OOD and retention performance. No additional code-task extension.

Each seed follows two policies through five cumulative updates. Score updates
3 and 5 in the same trajectory, retaining baseline and immediate scores. Update 4
is trained and archived but not scored. There is no separate, duplicate T=3 run.
The policies share the first update; thereafter they use constant LR 3e-5
(Native) or 1.5e-5 (lower-LR continuation). Both use eight distinct verifier-passing
corrections and 25 fresh-AdamW steps per update, rank 16/alpha 32 attention LoRA,
BF16, complete 1,024-token SFT targets, generation cap 512, frozen template date
`11 Oct 2026`, thinking disabled for Qwen, and the original 1,024 training prompts.

| Backbone | Fixed seeds | Entry point | Maximum GPU hours/seed |
| --- | --- | --- | ---: |
| Qwen3-4B | 2026110900–2026110903 | Reuse primary M0/M1 and canonical draw-0 M2 | 24 |
| Llama-3.2-3B | 2026111000–2026111003 | Fresh M0 | 24 |
| Qwen3-8B | 2026111100–2026111103 | Fresh M0 | 32 |
| Llama-3.1-8B | 2026111200–2026111203 | Fresh M0 | 32 |

The canonical Qwen draw and seed subset are fixed without inspecting its outcomes.
Reused parents are not counted as new independent replications. The other twelve
seeds are new. The shared-pretrained initialization plus seed controls harvesting,
LoRA initialization and optimization; it is not independent pretraining.

## Continuation and scarcity

At update 2, both policies use the same M1-generated donor batch and optimizer
seed. From update 3, each policy harvests from its own current model, using paired
random-number seeds and identical fixed harvesting budgets. This evaluates the
total policy trajectory, including endogenous data changes; it does not isolate
weight interference from donor-distribution effects.

Eight correction attempts per failed training problem; reject oversized SFT
targets before sampling. No duplicated problem within a batch, unverified fix,
dynamic batch shrinkage or replacement seed. Once a branch cannot supply eight
valid corrections, it is absorbing terminal: retain the same model at all
remaining generations and record availability=0. Initial scarcity freezes both
policies; update-2 scarcity freezes both at M1. No manufactured training updates.

Initial seeds match the primary recipe: harvest seed, batch seed+17, optimizer
seed+101. Update 2 uses harvest seed+10,000, batch +19 and optimizer +31. Later
generation t uses harvest seed+t*1,000,000, batch +19 and optimizer +31 for both
policies. Starting checkpoints, donor batches and cumulative chains are sealed.

## Scoring and interpretation

Use the frozen expanded pools: 1,319 GSM8K test, 500 MATH-500 and 1,024 MMLU test.
The same Hold model is the no-update baseline at every timepoint; deterministic
identity scores can be reused with a receipt. No score influences continuation,
recipe choice, seed inclusion or checkpoint selection. Seal the trajectory
before new final scoring. Original Qwen scores retain their original commitments
and checksums; do not rewrite them as newly generated evidence.

Report G1, Native gains versus Hold at T=3 and T=5, changes versus M1 at both
horizons, lower-LR versus Native at both horizons, and Native T5−T3. Joint
parent/shared-item bootstrap uses 5,000 replicates and preserves policy/item
pairing. Report ordinary 95% exploratory intervals and simultaneous intervals
over all eight endpoints and four fixed backbones. Four seeds can produce wide
intervals. Policies, generations and evaluation items do not increase seed N.
No causal scaling or universal long-horizon claim follows from this design.

## Execution and twenty-day envelope

Target window: November 9–29, assuming continuous A100 access. Scheduler delays
can extend calendar completion. The two accounts each permit one GPU, so total
continuous capacity is 960 GPU-hours over twenty days. The primary account
continues its fixed two-update cohort. The added account runs this extension
concurrently: 448 GPU-hours of protective trajectory ceilings plus four half-hour
compatibility ceilings, **450 GPU-hours maximum**, without retries or expansion.
These are limits, not runtime predictions. No new GPU job starts if its protective
limit no longer fits the remaining declared window. A failure is retained and
reported; it does not trigger an unbudgeted retry or silent cohort reduction.

Models are prepared and all tests run on CPU allocations while primary inference
continues. One additional-account GPU job runs at a time. CPU artifact inspection,
compression and statistics overlap subsequent GPU work. GPU jobs request 16 CPUs
and 64 GB host memory; CPU jobs use 2–8 CPUs, with Slurm enforcing the account's
32-CPU/256-GB aggregate cap. A one-CPU storage watcher is included in that cap.
Keep fixed generation batches 32/16 and the existing verifier semantics.

Reuse the original environment and Qwen4 weights read-only. Stage only the other
three pinned models, matching every official weight hash and the already pinned
public metadata sources. Reserve seven GB before downloads and two GB of scratch
before GPU work. The whole-account hard limit is 50 billion bytes; admission
stops at 48 billion including reservations. Retain every adapter losslessly,
verify its checksum and preserve failed logs. No login-node computation or
training, new forecasting fit, architecture sweep or automatic expansion.
