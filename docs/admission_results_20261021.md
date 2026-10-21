# Fresh admission development results

Reviewed October 21, 2026. All 16 prospective development parents completed:
four seeds per backbone/task stream, K=3 candidates and three native continuation
draws per candidate. The full fixed rendering context was used from the first
harvest onward. This is development evidence, not untouched confirmation.

## Execution and integrity

All elements of arrays **1081443–1081446** completed with exit 0. Accounting
shows all 16 GPUs overlapped in execution. The first allocation began October 20
at 13:40:59 IST; the last pilot finished October 21 at 02:33:37 IST. Automatic
CPU review **1081447** completed at 04:46:56 IST in 35 seconds, exit 0. The live
queue was empty when checked at 10:07:11 IST.

Structural inspection passed for every parent: 48 candidate labels (three distinct
compositions within each parent), 144 continuation observations, matched optimizer budgets, unique problems
within each batch, aligned target/OOD/retention outcomes and rendering provenance.
All continuation observations were available: **zero terminal draws**. Scarcity
did not prevent H2 comparisons in this collection. Initial training pools remain
math 1,024 and code 306; development evaluations are 64/64/256 examples.

Compute source:
`ef78aa1ed0565020b2fc727af62f8aaeac9cbf71ba7382f2aa839eb54da44156`.
Plan SHA256:
`251d833854405680b83d4d476ca1ae3032144f0b5771ed995a236a2ff5f6a895`.
Report SHA256:
`350818ce6d53fda8251bb30158e02cb340c7927d1e4f34f659abcab123d0309e`.
The downloaded report, fits, inspection and completion receipt were checked
against remote file hashes and the completion receipt's report hash.

Campaign:
`/scratch/11617/sujato_ts/third_eye_research/runs/admission_stage_b_ef78aa1ed056_env1`.
Local artifacts: `runs/deployment/evidence/admission_stage_b_20261021/`.

## Results

All values below are aggregate-utility **percentage points**, except coverage.
“Forced” means the immediate-selected candidate followed by its standardized
continuation; “hold” keeps the parent. This is not an isolated one-update effect.

| Policy or comparison | Mean pp | Parent-resampling 95% interval pp | Conditional scoring-item 95% interval pp |
| --- | ---: | --- | --- |
| Forced versus hold | -2.31 | [-3.82, -1.08] | [-3.91, -0.63] |
| Privileged admission proxy versus hold | +0.84 | [+0.28, +1.48] | [+0.004, +1.68] |
| Proxy versus forced | +3.15 | [+1.99, +4.70] | [+1.86, +4.43] |
| H1-trained gate versus hold | +0.16 | [-1.01, +1.15] | [-0.81, +1.17] |
| H2-trained gate versus hold | +0.13 | [-0.51, +0.60] | [-0.46, +0.72] |
| H1-trained gate versus forced | +2.47 | [+1.86, +3.09] | [+1.22, +3.66] |
| H2-trained gate versus forced | +2.44 | [+1.05, +3.99] | [+0.96, +3.89] |

The fresh, matched-context cohort reproduces a negative forced-update mean under
this protocol. Both separate uncertainty analyses exclude zero. They are
conditional parent/shared-item analyses, not a combined simultaneous population
test; the result does not establish that parameter self-updates generally fail.

The privileged proxy chooses admission from future observations and scores it
on other draws/items. Its +0.84pp point estimate clears the 0.5pp developmental
headroom target, but it is not deployable forecasting, a true oracle or an upper
bound. Its scoring-item lower bound is very close to zero. Positive proxy
headroom does not establish that the available features can learn it.

The learned gates beat forcing updates but **neither demonstrates improvement
over holding**. H1 coverage is 50%; H2 coverage is 28.125%. Simple protected
replay is +0.04pp versus hold, parent interval [-1.27,+0.90]. H1 minus replay is
+0.12pp [-0.26,+0.51]; H2 minus replay is +0.09pp [-0.98,+1.41]. Superiority over
replay is unresolved. The declared development criterion provisionally prefers
H1; H2's point estimate is slightly lower, and no restored forecasting claim is
supported. Failure to detect an advantage is not an equivalence result.

Forced retention is **+0.86pp**, parent interval [+0.47,+1.19]. Aggregate utility
loss must not be described as demonstrated retention degradation.

## Heterogeneity and scope

Each stream has only four parents; subgroup values are descriptive:

| Stream | Forced minus hold, mean pp | Privileged proxy minus hold, mean pp |
| --- | ---: | ---: |
| Llama code | -6.09 | 0.00 |
| Llama math | -7.68 | -0.84 |
| Qwen code | +2.67 | +3.01 |
| Qwen math | +1.87 | +1.19 |

The backbone asymmetry is substantial. The pooled negative mean must not hide
positive Qwen means, and the pooled proxy gain must not become a four-stream
admission success claim. A cheap stream-calibrated baseline is an important
exploratory diagnostic before attributing predictability to candidate features;
any such diagnostic is new development analysis, not a frozen primary result.

Scope remains two 3–4B backbones, LoRA, two-correction batches, K=3 and H2 at
initial parents. No recursive-policy or broad-model claim follows from these
16 initial states. The previously frozen historical/null results remain intact.

## Review decision and next work

The computed decision is `REVIEW_HEADROOM_POLICY_POWER_AND_COST_BEFORE_C`.
This is **not authorization to launch C**. Admission headroom is promising, but
learned discrimination, practical utility versus both constants/replay and
feasible confirmation power remain unmet.

At a 0.5pp effect, the fixed-stream residual planning sensitivity gives joint
superiority power of approximately 7.9%/14.5% for H1 and 8.9%/12.4% for H2 at
48/64 parents, far below the declared 80% target. The overall-parent sensitivity
is also weak. These are uncertain normal planning projections conditional on
the pilot fits, not validated power guarantees. Shared-item uncertainty and
refitting uncertainty are additional limitations. Do not launch the proposed
48–64-parent confirmation unchanged or simply spend the remaining allocation.

Next review should use existing development data to check backbone/stream
baselines, held-parent discrimination, feature support and the source of scoring
uncertainty. Declare any extra analyses as exploratory and run them under CPU
allocations. Freeze a useful rule and an affordable, powered confirmation design
before collecting C. If that cannot be achieved, retain the scoped empirical
result instead of promoting a near-hold gate as a novel forecasting success.
No new architecture sweep, scaling or recursive campaign is justified by B alone.

## Cost

Reconciled all 16 array elements and CPU review against final scheduler
accounting. Whole-stage B cost, including both earlier infrastructure attempts,
is **144.0331 SUs**, below its 192-SU ceiling. The persistent bounded-additions
ledger totals **231.8292 SUs charged/reserved** against the 1,200-SU cap; remaining
project-cap headroom is about **968.17 SUs**. This is project-budget headroom,
not a live TACC account balance. Earlier historical campaign costs are separate.
No job remains queued and no scientific expansion was submitted automatically.
