# Corrected crossed-correction experiment: October 18 review

Live check: October 18, 2026, 09:56 IST. All scheduled computation ran through
Slurm. The user queue is empty. No further experiment was launched.

## Execution and evidence

- Four-stream CUDA validation `1075257`: completed, exit 0, 3,363 seconds.
- Matrix `1075258_0` through `_23`: all 24 completed, exit 0.
- CPU review `1075259`: completed, exit 0, 32 seconds; approximately October 17,
  20:55 IST. Its status is `science_review_required`, with no missing tasks and
  `automatic_expansion=false`.
- The review validated all 216 fresh matrix outcomes, 72 historical exact adapter
  replays, fixed rendering context, baseline consistency, batch/checkpoint/seed
  provenance and matched optimizer budgets.
- Scope: eight previously inspected development parents, two backbones, math
  and code, three candidates, three continuation draws. The independent sample
  size is eight parents, not 216 cells.

Immutable source:
`a274ff920ac613d9ec78dad280e4dd153ea345bec1727cfbea2efbb929ab60df`.
Remote campaign:
`/scratch/11617/sujato_ts/third_eye_research/runs/crossed_corrections_a274ff920ac6`.
The complete compute-node review was downloaded without recomputation to
`runs/deployment/evidence/crossed_c1_20261018/report.json`.

## Predeclared results

Values below are percentage points of the configured aggregate utility,
converting the saved report's fractions by 100. Intervals are the report's
parent-level 95% confidence intervals.

| Comparison | Mean (pp) | 95% CI (pp) | Paired permutation p |
| --- | ---: | --- | ---: |
| Primary: native model/correction pairing surplus | -0.191 | [-0.611, +0.158] | 0.4766 |
| Native-future selection minus immediate selection | -0.651 | [-1.375, +0.065] | 0.1719 |
| Common-donor selection minus immediate selection | -0.439 | [-1.676, +0.602] | 0.6250 |
| Common-donor selection minus native-future selection | +0.212 | [-1.074, +1.546] | 0.8750 |

All declared secondary comparisons have Holm-adjusted p=1.0. Target, OOD and
retention pairing-surplus intervals also include zero. The future-aware
selectors use privileged continuation observations; they are not learned
forecasters. Their selection and scoring draws/items were separated as declared.

The experiment does **not** establish beneficial native pairing, an independent
future-selection advantage, or restored forecasting. It also does not prove
equivalence, universal unpredictability, absence of recipient/donor interactions,
or that the date defect explains earlier negative results. The signed pairing
contrast can cancel heterogeneous interactions. Small, inspected development
data do not support a confirmatory paper-wide claim.

## Budget and stopping decision

Reconciled completed-chain charges under the declared rates and 15-minute
minimums: about **24.65 SUs**, including CPU validation/preparation/review,
CUDA validation and **23.47 SUs** for the matrix. This replaces the chain's
conservative 196.44-SU reservation; it does not reset historical charges.
The persistent additional-experiment ledger now shows **87.63 SUs charged plus
reserved** against its 1,200-SU hard cap. The account display shows **1,997 SUs
available**; these are different accounting scopes.

The automatic review stop has been reached. This result does not justify another
forecaster architecture sweep or scaling. The next decision is a synthesis of
completed evidence and a defensible, testable claim. A positive forecasting or
native-feedback mechanism headline is currently unsupported. Any future
confirmation must specify its primary question and success criterion before new
outcomes, and address the historical rendering-context limitation.
