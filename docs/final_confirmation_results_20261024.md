# Final empirical confirmation: completed results

October24,2026. The fixed32-parent cohort completed without failed parent jobs:
eight independent seeds for each of Qwen3-4B/Llama-3.2-3B-Instruct and math/code.
The final analysis completed at02:56:18 IST. All416 checkpoint evaluations and
288 continuation draws completed; no continuation was terminal. No forecaster
was fitted. All scientific calculation ran in Slurm allocations.

The cohort, actions, selection/scoring separation and primary endpoint were
frozen before new outcomes. The collection source remains `43b5c215c6b0`.
Operational timing/concurrency changes did not alter scientific budgets or data.
Historical failures and unsuccessful designs remain preserved.

## Results and interpretation

All effects below are aggregate target/OOD/retention utility percentage points.
Intervals jointly resample independent parents and shared evaluation items.

| Policy | Future utility versus hold | Joint95% interval | Commit coverage |
| --- | ---: | --- | ---: |
| Hold | 0.000 | Reference | 0% |
| Forced immediate-selected update plus continuation | -1.760 | [-3.105,-0.420] | 100% |
| Protected replay | -0.273 | [-0.799,+0.206] | 21.875% |
| Fixed family admission: commit Qwen, hold Llama | -0.180 | [-0.916,+0.604] | 50% |

**The predeclared primary model difference replicated:** Qwen-minus-Llama
forced future utility was+2.801pp, joint95% interval[+0.266,+5.329]. This is a
difference between two fixed models under this protocol, not a causal effect of
model identity or a general claim about their families.

**Positive Qwen benefit did not replicate.** Its mean was-0.359pp; Llama's was
-3.160pp. Their simultaneous95% intervals were[-2.007,+1.439] and
[-5.527,-0.784] respectively. A difference in degradation is not a confirmed
positive-benefit/negative-harm crossover.

**The standardized continuation worsened the selected candidate's outcome.**
Forced future minus forced immediate utility was-1.776pp, simultaneous95%
interval[-2.876,-0.679], Holm-adjusted p=0.00120. Immediate selected-candidate
utility itself was approximately+0.016pp, joint95% interval[-1.319,+1.326].
This measured consequence supports distinguishing immediate repair from later
utility. It does not establish the mechanism of deterioration.

**The fixed admission rule is not a successful new method.** It improved on
forced updating by+1.580pp, simultaneous95% interval[+0.143,+2.962], Holm p=0.0140.
It did not beat hold or replay, missed the retention noninferiority criterion
and failed the full predeclared practical-advantage requirements. Replay versus
forced was+1.487pp, but its simultaneous interval crosses zero and Holm p=0.0576;
do not promote its unadjusted interval to an adjusted significant claim.

The supported paper is a scoped empirical study of self-update reliability:
verified corrections and immediate selection do not establish future utility,
forced candidate-plus-continuation updates underperform hold in this cohort,
and outcomes differ across the two tested models. The data do not support
forecasting success, universal self-update harm, an identified causal mechanism
or a generally useful learned admission controller.

## Artifacts, cost and completion

The local evidence directory is
`runs/deployment/evidence/final_confirmation_20261024/`: `report.json`,
`completed.json`, `findings.md`, `policies.csv`, `policy_utility.pdf`,
`policy_utility.png` and `budget_closeout.json`. The checked report hash is
`2037f908cd29778d60d1444a12f410fa7d79a86be3783b46768912963a9c1c43`.
The report retains provenance, stream means, all secondary comparisons and
separate parent/item/joint uncertainty. Stream results are descriptive.

Slurm-based accounting gives309.0019 SUs for complete C, including original
allocated validation/timing attempts. The bounded-additions ledger is542.5736
SUs used/reserved, leaving657.4264 under the1,200 cap. Historical original
campaigns are excluded. This is project accounting, not an account allocation
balance or a replacement for the service's eventual billing statement.

**Decision: experiments complete; freeze evidence and write.** There is no new
collection, architecture sweep, recursion or forecasting recovery queued. Remaining
work is the paper text, accurate comparison with prior work, artifact packaging,
and final checks of claims, tables and provenance. Any future scientific study
would require a separately declared question rather than chasing a favorable
outcome in this cohort.
