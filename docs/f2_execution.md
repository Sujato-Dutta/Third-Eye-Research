# F2 forecasting execution

Third Eye-Contrast is an isolated forecasting extension. The frozen
[F2 protocol](protocol_forecaster_f2.md) defines its inputs, objectives,
controls, split, selection rule and evaluation criteria. Original A2 source
and results remain preserved.

The model compares the three candidates from a common parent state and
predicts immediate and continuation consequences separately. Its H=2
prediction is their sum. Only information available before an update enters
the model; observed future consequences provide training supervision.

Nine local synthetic checks passed, covering candidate permutation,
within-state comparisons, future-outcome exclusion, grouping contracts,
finite gradients, save/reload, matched H=1 isolation, trajectory separation
and integration with the existing online runner. These checks establish
implementation behavior, not empirical forecasting improvement.

## Queued comparison

Vista CPU job **1057239** was submitted on October 8 at **13:45:57 IST**.
It requests one 72-CPU compute node through `gg,gh-dev` and runs the full
validation suite before fitting. Its pending wall-time request was shortened
from 30 to 15 minutes to improve backfill eligibility, without changing any
model fitting budget. No forecasting or training runs on a login node.

At **13:57:40 IST**, the job remained pending. The scheduler estimated
**15:16:54 IST**; this is an estimate, not an allocation guarantee.

All 15 fits use the fixed published snapshot: **97 training states,
40 validation states and 40 held-out test states**. F2 fitting and checkpoint
selection use training and validation only; no F2 test predictions or metrics
are computed in this job.

- Full H=2 and matched H=1: seeds 42, 43 and 44.
- Original independent GRU with the F2 objective and selection rule: H=2
  and H=1, seeds 42, 43 and 44.
- H=2 seed-42 ablations: no set comparison, no temporal auxiliary losses,
  and no comparative loss.

The predeclared primary model is `full_h2_seed42`. Another seed or ablation
cannot replace it because its validation score is higher. Gate 2 thresholds
remain unchanged. The job stops after its review report; it launches no
online trajectories or scaling.

## Evidence and versioning

Remote output: `runs/a2/forecaster_f2_v1` under the A2 workspace.
Local release manifest:
`runs/deployment/evidence/forecaster_f2_release_v1_20261008.json`.
Release manifest SHA-256:
`349f16a68ed7dbfa214aea148f90c3b25256fb33a4dc78a870c0c86d451c3f3d`.

Submission and scheduler receipts are preserved in monitor requests 375
and 379. Each fit verifies the frozen release and label snapshot, records
its architecture and normalization, and reloads the saved artifact for
prediction equivalence. The extension runner records its separate source
identity while using the existing online experiment implementation.

Improvement and scientific contribution remain unproven until the matched
comparison finishes. Passing Gate 2 would establish validation forecasting
quality under the stated criterion; recursive benefit and held-out
generalization still require their planned experiments.

## Completed comparison: October 8, 15:15 IST check

Vista job **1057239 completed successfully in 62 seconds**. Its compute-node
validation suite passed **126 tests in 33.39 seconds**, and all fifteen
predeclared fits finished. All exact artifact-reload checks passed.

The primary `full_h2_seed42` achieved validation future-H=2 Spearman
**0.14085** and informative top-1 **0.425** on forty nonconstant states.
**Gate 2 failed**, with unchanged thresholds of 0.30 and 0.45.
The original primary achieved 0.21585 and 0.50 on the same snapshot;
F2 did not improve that result. No full F2 H=2 seed passed Gate 2.
Test data remains unevaluated by F2. No online/scaling jobs were submitted.

All entries below report validation ranking of **future H=2** consequences.
H=1 models were fitted and selected using H=1 supervision; their H=2
columns are reporting only.

| Fit | Spearman | Top-1 | Selected epoch |
| --- | ---: | ---: | ---: |
| `full_h2_seed42` | 0.14085 | 0.425 | 3 |
| `full_h2_seed43` | 0.17500 | 0.375 | 92 |
| `full_h2_seed44` | 0.17835 | 0.475 | 1 |
| `full_h1_seed42` | 0.09085 | 0.375 | 26 |
| `full_h1_seed43` | 0.16250 | 0.425 | 75 |
| `full_h1_seed44` | 0.00335 | 0.350 | 44 |
| `independent_h2_seed42` | 0.28750 | 0.425 | 1 |
| `independent_h2_seed43` | 0.25000 | 0.375 | 7 |
| `independent_h2_seed44` | 0.26250 | 0.425 | 4 |
| `independent_h1_seed42` | 0.33750 | 0.400 | 7 |
| `independent_h1_seed43` | 0.16250 | 0.425 | 9 |
| `independent_h1_seed44` | 0.28750 | 0.450 | 5 |
| `no_set_h2_seed42` | 0.16250 | 0.400 | 4 |
| `no_temporal_aux_h2_seed42` | 0.14085 | 0.425 | 3 |
| `no_comparative_loss_h2_seed42` | 0.20000 | 0.400 | 15 |

Full result: `runs/deployment/evidence/forecaster_f2_review_v1_20261008.json`.
Artifacts, logs and frozen release are archived as
`forecaster_f2_evidence_v1_20261008.tar.gz`, with a durable Vista work copy.

The recovery controller completed and queued the three actual timeout
recoveries (`1057197`, `1057269`, `1057271`). The final audit/forecast job
`1057272` waits for successful completion of all three. At this check the
GPU recoveries remained pending for scheduler priority.

This comparison is preserved as negative evidence. It does not establish
an implementation defect or disprove the research direction; it provides
no support for claiming forecasting improvement from the new mechanism.
Further development must address predictive inputs and generalization
without changing the frozen criteria or selecting a favorable seed.
