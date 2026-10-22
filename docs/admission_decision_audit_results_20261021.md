# Admission decision audit: completed exploratory review

Reviewed October 21, 2026 after downloading the completed compute-node outputs.
The [declaration](admission_decision_audit_20261021.md) remains unchanged. These
are exploratory analyses of the already inspected 16-parent B cohort, not new
independent confirmation, a promoted controller or restored forecasting success.

## Execution and provenance

Validation **1085722** completed with exit 0 in 102 seconds: 186 tests plus lint.
CPU analysis **1085723** completed with exit 0 in 41 seconds, at **11:57:38 IST**.
All saved-adapter checks, model fitting, bootstraps and planning calculations
ran inside the allocated CPU job. Login access only staged files and inspected
scheduler, hashes and accounting metadata. No GPU job was added.

The audit exactly reproduced the original B gate hyperparameters, predictions
and admission decisions before evaluating the new diagnostics. All 16 parents
retain K=3 complete candidates and nine continuation outcomes per parent, with
zero terminal draws. Original B evidence was preserved.

Source: `6f87e4f482b8791a71ac14090e9b14e04e9489e4849b64c72132cc5e08269ef4`.
Report SHA256:
`8bf8b1c541ce5327dfd019c05a7fee96e3ff763cbe086aff06dcc5ec9ed36398`.
The local downloaded report matches its completion receipt and remote hash.
Artifacts: `runs/deployment/evidence/admission_decision_audit_20261021/` contains
`report.json`, `predictions.json`, `completed.json` and `accounting.json`.

## What the audit says

Utility values below are percentage points relative to holding the parent.
Each CV prediction excludes its test parent's future outcomes; training and
threshold selection use selection-half items, with scoring halves for outcomes.
Both item orientations are averaged within the 16 parents.

| Policy | Whole-parent CV | Held-seed-group CV |
| --- | ---: | ---: |
| Forced immediate-selected candidate plus continuation | -2.307 | -2.307 |
| Protected replay | +0.037 | +0.037 |
| Original H1 ridge | +0.159 | -0.034 |
| Original H2 ridge | +0.132 | +0.324 |
| Backbone mean, H1 labels | -0.704 | -0.704 |
| Backbone mean, H2 labels | **+1.135** | **+1.135** |
| Backbone/task stream mean, H1 labels | +0.034 | -0.186 |
| Backbone/task stream mean, H2 labels | +0.753 | +0.799 |
| Stream mean plus candidate-feature residual, H1 | -0.503 | -0.224 |
| Stream mean plus candidate-feature residual, H2 | -0.118 | +0.728 |

The H2 backbone baseline admits the immediate-selected Qwen update and holds
Llama across both orientations and CV schemes: 50% admission coverage. Its
whole-parent conditional interval is **[+0.564, +1.727] pp** versus hold, and
its mean advantage over forced updates is **+3.442 pp**. However, directly
crossing shared-item and parent resampling gives **[-0.011, +2.360] pp** versus
hold. This wider interval includes zero. The corresponding retention interval
is [-0.350, +0.806] pp; this conditional diagnostic is not the planned
confirmatory retention test.

The H2 stream baseline has a crossed interval of **[-0.160, +1.693] pp** versus
hold. Global H2 means hold everything in both CV schemes; pooling all backbones
therefore hides useful heterogeneity. Task-only H2 means mostly hold as well.

This is evidence worth investigating for **context-level admission calibrated
with future outcomes**. It does not establish candidate-specific prediction,
transfer to unseen backbones, a novel algorithm, or prospective H2 superiority.
The baseline was evaluated after inspecting B and among multiple exploratory
methods. Matching actions across CV schemes is robustness within this cohort,
not two independent replications.

Candidate features did not add consistent value over context: residual H2 minus
stream H2 is -0.871 pp in whole-parent CV and -0.071 pp in held-seed-group CV.
The residual diagnostic varies with the split, and the original learned gates
still have no demonstrated useful advantage over hold/replay. No architecture
sweep follows from this result.

The forced-update crossed interval remains negative, **[-4.684, -0.124] pp**.
That scoped finding concerns candidate-plus-continuation utility under this
protocol. It must not be generalized to all models or called demonstrated
retention degradation; Qwen and Llama have different mean outcomes.

## Precision and cost review

Shared evaluation items remain a material source of uncertainty. Increasing
parent count without increasing independent scoring information is insufficient
in the reported conditional projections. For the H2 backbone rule, hypothetical
N=64 with four times as many independent same-population items gives about
4.1% joint superiority power for an assumed +0.5 pp effect against each of the
two comparisons, or 45.2% for +1 pp each, below the declared 80% target.
These are fixed-action empirical planning sensitivities, not validated power:
they exclude refitting, new-domain uncertainty and separate retention criteria.

Measured pilot maxima plus 25%, rounded to 15 minutes, imply the following
**base GPU reservations before additional scoring, CPU checks or retries**:

| Fresh parents | Conservative base SUs |
| --- | ---: |
| 32 | 404 |
| 48 | 606 |
| 64 | 808 |
| 96 | 1,212 |

Thus the original 48–64-parent C proposal does not fit its 560-SU ceiling with
this conservative reservation. Dropping optional recursion could free budget,
but does not solve precision or make extra inference free. Code final OOD is
HumanEval, whereas development OOD is MBPP validation; larger-suite precision
cannot be assumed equivalent. Historical final-item exposure is unresolved.
This audit neither read nor generated final predictions.

Final scheduler reconciliation charges **0.165 SU** for validation plus audit,
including the CPU minimum billing. B development including this audit totals
**144.1981 SUs**, below 192. The bounded-additions ledger is **231.9942 SUs**
charged/reserved, with **968.0058 SUs** remaining under the 1,200-SU project cap.
These numbers are not the TACC account balance or total historical project cost.

## Decision

Retain `STOP_FOR_POLICY_PRECISION_AND_COST_REVIEW_NO_C_RELEASE`.
No confirmation or recursive campaign is queued by this audit.

The next concrete planning work is to establish scoring-item provenance and
benchmark additional independent scoring on a small allocated compute job,
then price a balanced design with optional recursion removed first if needed.
The backbone H2 baseline is a candidate to freeze alongside H1 calibration and
replay, rather than claiming feature-based forecasting has recovered. Any
policy promotion, endpoint/design change and reservation transfer must be
documented before prospective outcomes. If practical superiority and adequate
precision cannot fit the cap, retain the scoped empirical result and stop
expansion. The audit has identified a useful lead, not cleared confirmation.
