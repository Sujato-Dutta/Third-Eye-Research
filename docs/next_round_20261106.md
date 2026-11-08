# Next-round execution: positive regimes and matched continuations

November 6, 2026. Additional Vista hard limit: **200 SUs**, including validation,
minimum billing, allocated failures and retries. The previous completed study
and its Overleaf archive remain unchanged.

**November 8 allocation change:** new work moved to DGX at the user's request.
Vista array `1125744` was cancelled before starting. The current job chain,
one-A100 limit, storage checks, and automatic eight-parent review are recorded
in [DGX execution](dgx_next_round_20261108.md). The Vista tables below are retained
as execution history; no new Vista GPU work remains queued for this round.

## Submitted jobs

Saved-analysis source: `91176162e2017eb82ccfe0e9b125258f486a7acb7374502a580c05cb838a84c5`.
Recipe-pilot source: `149c4aac3ca35b07f9d3e4c333476f89a1282fa6e42a00c50e7d2ac83d99fa25`.
The archive checksum also binds the validation tests.

| Site | Job | Work | Dependency | Maximum Vista SUs |
| --- | --- | --- | --- | ---: |
| Vista | 1125713 | Original CPU validation; packaging failure preserved | none | 0.33 |
| Vista | 1125719 | Full CPU validation with missing static test fixtures restored | 1125713 ends | 0.33 |
| Vista | 1125714 | Original manifest preparation; malformed source reference detected | 1125719 passes | 0.33 |
| Vista | 1125715 | Saved-output sensitivities, 56 parents | 1125719 passes | 1.32 |
| Vista | 1125716 | Cancelled before starting; zero elapsed time confirmed by accounting | none | 0 |
| Vista | 1125728 | Dataset-loader validation; temporary-directory setup failure preserved | none | 0.33 |
| Vista | 1125735 | Full validation with temporary directory created | none | 0.33 |
| Vista | 1125729 | Preparation found an empty boxed source reference | 1125735 passes | 0.33 |
| Vista | 1125730 | Cancelled before starting; zero elapsed time confirmed | none | 0 |
| Vista | 1125743 | Input preparation, cached-model checks, complete suite | none | 0.33 |
| Vista | 1125744, array 0--1 | Two six-hour development/cost pilots | 1125743 passes | 12.00 |
| DGX | 34652 | Original CPU validation; packaging failure preserved | none | 0 |
| DGX | 34655 | Full CPU validation with static test fixtures restored | 34652 ends | 0 |
| DGX | 34653 | Saved-output sensitivities, eight Qwen-8B parents | 34655 passes | 0 |

Latest verified state: **1125743 completed successfully**. Outcome-blind manifest
preparation, cached-model checks, all **253 tests**, and lint passed. Both
`1125744` tasks are pending scheduler priority, with dependencies satisfied.
All scientific work ran on allocated compute nodes.

Live check, November 6 at approximately 23:08 IST: both `1125744` tasks remain
pending for **Priority**, with no remaining dependency and zero GPU runtime.
The scheduler reports no estimated start time. The `gh` partition has no idle
nodes in this snapshot. No new pilot outcomes have been generated since the
completed saved-data diagnostics.

Live check, November 7 at 08:41 IST: both `1125744` tasks are still pending
for **Priority**, with zero runtime and no estimated start. Six `gh` nodes are
completing other work, but none is idle in this snapshot. No pilot output
directory or new scientific outcome exists yet; the allocation wait continued
overnight.

The current conservative Vista reservation is **15.63 SUs**, not actual usage.
The original package omitted `experiments/configs/*.json`, causing eight
missing-file test failures on DGX (242 tests passed). Both complete suites
were resubmitted with those fixtures, without changing scientific source.
The supplemental fixture archive is bound by SHA-256
`7402c6d072e516374dacec2fd410d7c0bb3c5bfae8b52331cea718d180636a39`.
Both repaired suites passed all 250 tests and lint. Failed logs and validation
receipts remain available.

Preparation then detected a MATH training solution without a boxed reference.
The loader now excludes and records missing, empty, or unbalanced references before deterministic
sampling; no model outcomes were used. A regression test covers that case.
The replacement source archive includes all static configuration fixtures.
The validation launcher also now creates its nested temporary directory before
pytest. Its failed setup receipt is retained; the retry verifies every extracted
file against the source archive before running the full suite.
The intermediate repair passed all 251 tests and lint; its dataset preparation
then exposed empty boxed references. Regression coverage now includes missing,
empty, and unbalanced boxes. Final input preparation precedes the complete suite
in one allocated CPU job, so input errors are found before test reruns.
New manifests and pilot outputs use `regime_clean/`, preserving failed
preparation artifacts. New GPU jobs depend on both successful preparation and validation;
the cancelled GPU reservation was released only after zero runtime was confirmed.
Saved-data analyses completed successfully for all 64 parents: 56 on Vista and
eight Qwen-8B parents on DGX. Reports retain strict and lenient estimates,
selected and uniform-candidate paired intervals, and effective adapter norms.
No confirmation, mitigation, remaining six development parents, or full
historical benchmark rescoring is submitted automatically. The two GPU pilots
run concurrently when allocated and remain queued independently of local SSH.
Per-site `runs/next_round_20261106/ledger.json` records submission receipts and
conservative reservations; failed allocated work remains chargeable.

## Completed saved-output sensitivities

All 64 parents passed artifact and retained-adapter integrity checks. On the
original 32-parent cohort, newly computed descriptive estimates are:

| Paired continuation change | Strict mean (pp) | Strict 95% interval | Lenient mean (pp) | Lenient 95% interval |
| --- | ---: | --- | ---: | --- |
| Development-selected candidate | -1.776 | [-2.544, -1.010] | -1.773 | [-2.536, -1.008] |
| Uniform candidate weights | -1.373 | [-1.893, -0.879] | -1.359 | [-1.870, -0.870] |

These are exploratory ordinary 95% joint parent/shared-item bootstrap intervals,
not replacements for the original confirmatory interval family. Lenient
rescoring changes only numeric GSM8K final-answer extraction. The observed loss
survives this specific parsing sensitivity; this does not rule out other format
effects or identify a mechanism. Effective per-layer first, second-increment,
and cumulative adapter norms are retained for all candidates and draws.

Verified reports and receipts are downloaded under
`runs/deployment/next_round_20261106/results/`. The previous paper remains frozen.

## First commitment: at most 55 SUs

1. Run saved-output diagnostics on allocated CPUs: a frozen lenient GSM8K parser,
   effective LoRA update norms, and uniform-candidate paired continuation
   intervals. These require no new GPU inference, but CPU allocations are charged.
2. Freeze expanded development and evaluation manifests before new outcomes.
   Reuse the existing training problems and official dataset revisions.
   Pilot recipe selection uses development data only. Evaluation pools already
   scored historically are not described as newly unseen.
3. Run two bounded cost pilots, one math parent per backbone. They are also the
   first two development runs. Measure harvesting, training, all recipe scores,
   native continuations, matched M0-direct arms, loading and verification.
4. Price the remaining development runs before submitting them. Complete four
   fixed development parents per backbone only when charged plus reserved cost
   fits the 55-SU stage and 200-SU overall limits.
5. Freeze the recipe/model decision using development results. Stop new research
   collection if no eligible positive regime appears. Do not substitute seeds,
   lower the batch size, or read final outcomes to rescue a regime.

The original suggestion of 15 SUs for scoring every saved checkpoint on full
benchmarks is an unverified estimate. The initial scoring allowance is for
necessary checkpoints and measured timing, not all 832 historical checkpoints.
Full GSM8K/MATH evaluation and larger MMLU sets must be priced per checkpoint.
No complete 16--20-parent confirmation or sub-1-pp half-width is promised before
cost and precision are measured.

## Development design

Backbones: Qwen3-4B and Llama-3.2-3B-Instruct, math only. Retain their exact
model revisions, fixed templates, rank-16 attention LoRA, eight correction
attempts, maximum generation length 512 and optimizer-reset semantics.
Historical training and final artifacts are never overwritten.

Four recipes form a fixed fractional grid, not an exhaustive architecture search:

| Recipe | LR | Optimizer steps | Unique corrections |
| --- | ---: | ---: | ---: |
| r0 | 1e-5 | 10 | 8 |
| r1 | 3e-5 | 25 | 8 |
| r2 | 1e-5 | 25 | 32 |
| r3 | 3e-5 | 10 | 32 |

One parent correction pool is shared across the four recipes. Deterministic
problem-identity ordering and fixed seeds determine batches; examples are unique
within a batch. If a recipe lacks its required corrections, it is unavailable.
No repetition or silent batch-size reduction is permitted.
Verified corrections must also fit the frozen 1,024-token complete SFT target;
overflowing targets are counted and excluded before batch construction, never
silently truncated.

Development seeds are 2026110600--2026110603 for Llama and
2026110700--2026110703 for Qwen. The two first seeds are cost pilots.
Candidate evaluation uses an expanded development suite, disjoint from update
training and final evaluation. The goal is mean equal-role G1 >= +1 pp, with
mean retention change >= -0.5 pp, across all four development parents of a
model/recipe. This is a screen, not evidence of confirmed positive benefit.
Among eligible pairs choose the largest development G1; exact ties follow
model name, then recipe ID. If no pair qualifies, stop.

Cost pilots additionally use fixed r0 for three native continuation draws and
three M0-direct updates on the exact same donor batches. These small development
timings are not final efficacy evidence or additional confirmation parents.
M0-direct begins from the original parent, uses a fresh optimizer, and has
the same donor problems, solution texts, optimizer seed, LR and steps as its
paired M1 continuation. Its utility is measured relative to M0.

## Prospective confirmation, conditional on a positive development screen

The remaining allocations are ceilings:

| Work | Maximum additional SUs |
| --- | ---: |
| Expanded evaluation/timing, CPU analysis and validation | 15 |
| Development/cost pilots | 40 |
| Fresh matched confirmation | 80 |
| One mitigation comparison | 30 |
| Contingency | 35 |
| **Total** | **200** |

Freeze confirmation N, seeds, item counts, endpoints and mitigation choice after
development, before inspecting confirmation outcomes. Target 16--20 fresh
parents and three native draws. If their measured reservations cannot fit,
report the priced shortfall rather than quietly reducing repeats or claiming
an underpowered cohort is powered.

Confirmation reports G1, native G2-G1, and native G2 versus M0-direct.
Positive G1 requires a confidence interval above zero; a development screen
alone is insufficient. Intervals jointly resample independent parents and
shared evaluation items, preserving native/direct pairs and candidate nesting.
Use simultaneous intervals for the declared primary family.

The variance estimate determines whether more items or more parents buys
precision. A half-width below 1 pp is a design target, not a guaranteed outcome.
Native correction scarcity preserves M2=M1 and is reported explicitly.
The direct arm uses the same no-update status for an empty donor batch.

Choose exactly one mitigation on development evidence: LR decay, fewer steps,
or training-only replay mixing. Compare it on the same fresh parent/donor
structure; do not select it using final outcomes.

## Interpretation and execution

- Lenient rescoring is a sensitivity analysis, not a replacement for original
  results. It may show that formatting contributes to a measured loss.
- Compare effective weight updates, not raw LoRA-factor distances:
  update 1 is W1-W0, update 2 is W2-W1, and cumulative drift is W2-W0.
  Norm associations do not establish a causal explanation.
- A null or negative result remains a valid result. No acceptance probability
  or desired sign is an execution criterion.
- All fitting, scoring, tokenization, adapter reading and statistics run under
  Slurm on compute nodes. Login sessions only stage files, query the scheduler
  and retrieve reports. No new forecaster or architecture sweep is included.
