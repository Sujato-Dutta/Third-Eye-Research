# Deadline signal experiments and direction decision

Request started October 8, **15:44:34 IST**. Both compute jobs completed by
**16:03:59 IST**, within twenty minutes. Reports, saved model artifacts,
statistical comparisons and plots were preserved within the thirty-minute
decision window. This is an empirical-analysis review, not a success claim
for the architectural proposal or a guarantee of publication.

## Completed experiments

- Vista CPU job **1057337**: **133 tests passed**, then 36 fixed independent-GRU
  fits and 24 fixed ridge fits. Three targets: immediate H1, future H2, and
  continuation increment H2-H1. Four nested whole-trajectory training budgets
  and three neural fitting seeds. Runtime **75 seconds**, including validation.
- Vista CPU job **1057342**: **two confirmation checks passed**, then one frozen
  held-out confirmation of the nine full-training GRUs, six deterministic
  ridge replays and two simple controls. No neural retraining, checkpoint
  selection or parameter tuning from test. Runtime **34 seconds**.
- Label decomposition, candidate margins, strict ranking reversals, immediate
  oracle regret, per-head/model/task/generation breakdowns, feature variation,
  tie-adjusted controls, trajectory bootstrap intervals, paired comparisons
  and Holm adjustment across three predeclared tests.

Neural fits used 97 training states in twenty trajectories. Development and
held-out confirmation each contained forty states in eight trajectories.
F2 and original A2 source/results remain preserved. Neither job allocated a
GPU, and no scientific computation ran on a login node.

Protocols: [diagnostic](protocol_signal_diagnostic_v1.md) and
[confirmation](protocol_signal_confirmation_v1.md). The confirmation is an
existing held-out trajectory population, not newly collected replication:
original label reports already exposed aggregate Gate 1 over the published
population, and earlier original forecasting jobs generated test metrics.
These facts must be disclosed. Test forecasting results cannot now be used
for iterative method changes and later described as untouched confirmation.

## Findings

Mean within-state Spearman across the three full-training GRU fitting seeds:

| Prediction target | Development | Held-out confirmation |
| --- | ---: | ---: |
| Immediate H1 | 0.41027 | -0.01250 |
| Future H2 | 0.26667 | 0.03110 |
| Continuation increment | 0.03363 | -0.12887 |

Corresponding held-out top-1 rates were 30.0%, 35.0% and 29.17%. H2's
tie-adjusted chance was 34.17%. The fixed lowest-pre-update-NLL heuristic
scored held-out H2 Spearman 0.19665 and top1 42.5%; the full-training neural
predictor does not demonstrate an advantage. Relative ridge features did
not rescue H2 prediction. A favorable smaller training subset or seed cannot
replace the full-training primary result.

The development H1 learning curve rises with training trajectory count
(0.1689, 0.2478, 0.2728, 0.4103). H2 is non-monotonic
(0.2833, 0.2583, 0.2417, 0.2667). The held-out results prevent concluding that
only continuation is difficult: even immediate prediction generalizes poorly.
All three paired horizon contrasts fail Holm-adjusted 0.05 significance in
both cohorts. In particular, held-out H1-own minus H2-own has mean -0.0436
and 95% trajectory interval [-0.2444, 0.1531]. A clean learnability hierarchy
is not established.

The more stable phenomenon is the mismatch between observed update rankings:

| Metric | Development | Held-out confirmation |
| --- | ---: | ---: |
| Strict H1/H2 winner reversal | 60.0% | 57.5% |
| Reversal 95% trajectory interval | 45.0–72.5% | 42.5–72.5% |
| Immediate-oracle future utility regret | 0.01751 | 0.01629 |
| Regret 95% trajectory interval | 0.01084–0.02464 | 0.01136–0.02080 |
| Future winner margin within one target-item utility unit | 22.5% | 25.0% |
| Exact future winner ties | 2.5% | 2.5% |

Regret is the gap between the observed H2-best candidate and a candidate
chosen by observed H1-best, averaging ties. It is an equal-weight utility in
accuracy-fraction units, not measured online policy improvement. Its point
value corresponds to 1.63 percentage points of aggregate utility on held-out
states. The future oracle is privileged, and maximizing over noisy finite
evaluation scores can inflate its apparent advantage; the descriptive gap
does not establish an attainable policy benefit or intrinsic-noise ceiling.

Continuation variance exceeds immediate variance in the utility decomposition,
with negative immediate/continuation covariance in training, development and
held-out cohorts. This describes ranking disruption under this protocol;
the algebra alone does not identify its cause. Seven encoded features have
no within-parent variation. Some are shared context by design, and verifier
pass rate is constant because only passing corrections survive. Removing
them cannot be claimed to fix the forecaster without a new independent study.

All audited A2 branches have continuation available. There is no A2
terminal/non-terminal contrast supporting a saturation finding.

## Split limitation

The original split is grouped by whole trajectory but is not stratified by
stream. Training contains 30 Llama-code, 27 Llama-math, 25 Qwen-code and 15
Qwen-math states. Development contains 15, 10, 5 and 10 respectively. Held-out
contains **zero Llama-code**, five Llama-math, fifteen Qwen-code and twenty
Qwen-math states. The missing Llama-code coverage limits four-stream held-out
claims. Do not rearrange this split after seeing its results.

The development-to-test drop can reflect limited training, adaptive development
selection, stream imbalance, trajectory shift or insufficient inputs. The
current experiments do not isolate those explanations. Report confidence
intervals and all cohorts, including the unfavorable confirmation.

## Direction decision

**Pursue an empirical analysis of persistent-update ranking mismatch and
forecasting generalization.** A successful new-architecture narrative is not
supported. A narrative asserting that H1 is reliably learnable while only
continuation is unpredictable is also not supported. Gate 2 remains failed;
there is no online/scaling release and no favorable-test replacement of its
development decision.

The observed ranking mismatch is a possible contribution, with concrete
observed selection regret and a reproducible outcome dataset. Main-conference
strength remains unproven: it needs an explanation or useful lesson plus
robust independent evidence, beyond failed architectures. Narrow claims to
the tested LoRA self-update protocol and covered model/task populations.

The highest-priority additional evidence is a **balanced, predeclared
replication cohort** and, if stochastic-instability claims are made, bounded
continuation repeats on saved checkpoints. Eight balanced new T5 trajectories
would provide forty new states across all four streams: approximately 174 GPU
node-hours and 24–40 hours after allocation, with queues and long tails adding
time. This cannot finish in a thirty-minute decision window. No such new GPU
jobs were launched as part of this quick study. The existing three checkpoint
recoveries and their dependent full-data audit remain queued.

Freeze the tested methods and analytical comparisons. Begin writing protocol,
resource, split, outcome and limitation sections now. Do not add more models,
architectures or thresholds to turn these results into a positive claim.
If balanced replication cannot complete before the freeze, either submit a
clearly scoped empirical paper if its contribution is adequate, or defer;
additional GPU usage cannot guarantee ACL Main acceptance.

## Reproducible evidence

Results are isolated remotely in `runs/a2/signal_diagnostic_v1` and
`runs/a2/signal_confirmation_v1`. Both archives have durable Vista work copies
and local copies under `runs/deployment/evidence/`. Artifact weights, metadata,
normalization and source are hash-bound. Validation reload equivalence was
checked before held-out inference. Preserve the diagnostic and confirmation
protocols and all predeclared seed/control results.

- `signal_diagnostic_v1_review.json`
- `signal_diagnostic_v1_signal_decomposition.json`
- `signal_confirmation_v1_review.json`
- `signal_confirmation_v1_signal_decomposition.json`
- `signal_diagnostic_evidence_v1_20261008.tar.gz`
- `signal_confirmation_evidence_v1_20261008.tar.gz`

Plots: [development learning curves](../runs/deployment/evidence/signal_figures_v1/learning_curves.png)
and [paired confirmation intervals](../runs/deployment/evidence/signal_figures_v1/paired_confirmation.png).
SVG versions are saved alongside the PNGs. The learning-curve shading is the
range across fitting seeds, not a statistical confidence interval.
