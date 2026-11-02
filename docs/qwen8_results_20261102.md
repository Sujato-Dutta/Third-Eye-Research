# Qwen3-8B complete scale-sensitivity results

November2,2026. All eight parents (four code/four math) and CPU analysis29453
completed successfully. The report contains104 scored checkpoints and72 native
continuation draws. All eight downloaded parent completion and lossless-retention
receipts passed source/plan checks; none of the draws were terminal.

Source SHA256:
`8777e39cac291e8cdcee09d41e7a1b54d9d4ef1ccec78caf4429f71ad9ab7fde`.
Plan SHA256:
`66e9eb5ad25571a2e0820303d9e806a7d133fbe9eb4ca7676e16a8d005b408cc`.
Report SHA256:
`3df09f8249a26282ed0dbf2d0a436aacad9c873ddb6cb0aa80a157969311d0d7`.
The downloaded report matches the passed analysis receipt and frozen source/plan.
Local evidence: `runs/deployment/evidence/dgx_scale_20261102/`.

## Aggregate results

Utility is the equal-weight average of target, OOD and retention accuracy.
All entries are percentage points. H1/H2 gains are relative to the parent;
change is H2 minus H1. Selection uses only development outcomes, with uniform
ties. Intervals jointly resample parents within task, shared exact scoring-item
identities and nested continuation draws. The two model-specific scale primaries
use97.5% intervals for joint95% coverage. Other endpoints are exploratory.

| Endpoint | Mean |97.5% interval |
| --- | ---: | --- |
| Selected H1 gain | -0.439 | [-2.262, +1.465] |
| Selected H2 gain | -1.120 | [-2.938, +0.719] |
| **Selected H2 minus H1 (primary)** | **-0.681** | **[-1.914, +0.629]** |
| Uniform H1 gain | -0.749 | [-2.417, +0.931] |
| Uniform H2 gain | -1.550 | [-3.235, +0.143] |
| Uniform H2 minus H1 | -0.801 | [-1.673, +0.230] |
| Replay H2 gain | -0.477 | [-1.733, +0.442] |

## Role decomposition

| Role | Selected H1 | Selected H2 | H2 minus H1 |97.5% change interval |
| --- | ---: | ---: | ---: | --- |
| Target | -0.342 | -1.237 | -0.895 | [-3.793, +1.913] |
| OOD | +0.049 | -0.537 | -0.586 | [-3.223, +2.507] |
| Retention | -1.025 | -1.587 | -0.562 | [-1.921, +0.688] |

## Interpretation

The continuation-change point estimate is negative, but its declared primary
interval crosses zero. This cohort does not establish additional harm at8B,
equivalence between horizons, or a beneficial effect of larger model size.
The selected H2-versus-hold contrast is also unresolved. Uniform H2 has a
nominal95% interval below zero, but its97.5% interval crosses zero and it is an
exploratory endpoint; do not present it as a confirmed primary finding.

This is a balanced eight-parent sensitivity cohort, with four seeds per task.
It has less precision than the main16-parent/model confirmations. It runs on
DGX A100 hardware; the smaller-model confirmations and pending Llama8B cohort
use Vista. Hardware, model family and training history remain confounded in
cross-model comparisons. The original32-parent confirmation and SmolLM result
retain their own declared analyses; this follow-up does not replace those tests.

The defensible current picture includes confirmed continuation losses in the
main smaller-model cohorts and an inconclusive Qwen8B extension. The final
Llama8B cohort remains pending Vista allocation. Keep that distinction in the
paper and avoid a universal or causal scaling claim.

## Runtime and preservation

Code parents took7h13m--8h01m; math parents took12h35m--16h38m. Analysis took13s.
All labels, batches, item outcomes and hash-verified lossless adapter archives
are retained on DGX. These jobs consumed no Vista SUs. DGX currently has no
active Third Eye job after this block and its analysis completed.
