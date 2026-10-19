# Stage A: corrected admission evidence

Reviewed October 18, 2026, approximately 17:53 IST. This is exploratory evidence
from eight previously inspected initial parents: two in each backbone/task stream.
No untouched confirmation or new GPU collection has been completed at this stage.

## Execution

Validation **1078453** completed successfully in 72 seconds: **171 tests passed
in 56.14 seconds**, followed by lint. CPU analysis **1078454** completed in
30 seconds, exit 0, with no failure artifact. Slurm times indicate the analysis
finished approximately 14:26:43 IST. Jobs reached the declared scientific stop;
no next-stage job was submitted automatically.

Source: `c8c37ae448f5531f146807c2e007f3a842e18b00de6a3ca26ef5d84fbc4b418e`.
Report SHA256: `10d11cb1791ff7a8b22fce0603f7e1f80b4b41c858846e38cff7e934eaad24a8`.
Campaign: `/scratch/11617/sujato_ts/third_eye_research/runs/admission_stage_a_c8c37ae448f5`.
Local receipts, report, fits and CSVs: `runs/deployment/evidence/admission_stage_a_20261018`.

## Main estimates and uncertainty

Values are aggregate-utility percentage points. The parent intervals stratify
resampling within the four fixed streams, with only two parents per stream;
they are conditional on shared finite evaluation items and fitted CV policies.
The separate item intervals resample common scoring pools together while keeping
decisions/models/checkpoints fixed. Neither is a combined population interval.

| Comparison | Mean pp | Parent-resampling 95% interval pp | Conditional scoring-item 95% interval pp |
| --- | ---: | --- | --- |
| Forced immediate-selected candidate plus standardized continuation versus hold | -1.18 | [-1.75, -0.59] | [-2.76, +0.38] |
| Privileged independently scored admission proxy versus hold | +0.92 | [+0.40, +1.44] | [+0.14, +1.74] |
| Admission proxy versus forced | +2.10 | [+1.67, +2.52] | [+0.74, +3.49] |
| H1-trained gate versus hold | -0.10 | [-0.50, +0.30] | [-1.20, +1.03] |
| H2-trained gate versus hold | +0.32 | [+0.18, +0.46] | [-0.10, +0.76] |
| H2-trained gate versus forced | +1.50 | [+0.93, +2.05] | [+0.01, +3.02] |

Simple protected replay has mean **+0.52pp versus hold**, parent interval
[-0.10,+1.14]. H2-trained gate minus replay is **-0.20pp**, parent interval
[-0.68,+0.28]. No superiority over replay is established. The H2 gate admits
12.5% of the candidate tie/fold-weighted decisions; the privileged proxy admits
41.7%. A positive conditional parent interval must not obscure the learned
gate's item interval crossing zero, limited coverage or inspected development data.

The proxy uses future outcomes to choose admission and independent draws/items
to score it. It is not a deployable learned method, ideal oracle or upper bound.
Its mean exceeds the 0.5pp development headroom target, warranting a bounded
fresh-development review, not confirmation release.

## Heterogeneity and mechanism limits

| Stream | Forced future utility versus hold, mean pp | Parents |
| --- | ---: | ---: |
| Llama code | -4.67 | 2 |
| Llama math | -4.14 | 2 |
| Qwen code | +3.63 | 2 |
| Qwen math | +0.47 | 2 |

These subgroup estimates are descriptive. The proxy's benefit is concentrated
in Qwen code, so no four-stream admission success is established. Forced
retention has a positive point estimate (+1.09pp); do not frame the aggregate
utility loss as universally caused by retention degradation.

Cross-draw residual interaction agreement is negative at the point estimate and
its interval includes zero. `pooling_sensitivity_eligible_development_only=false`:
the predeclared condition for the extra pooled-control experiment has not passed.
No spectral, third-model or H=3 work is justified by these outputs.

Although future evaluation rendering is fixed, the initial candidate checkpoints
and their correction histories come from historical collection. This is not a
fully fresh, fixed-rendering recursive replication. Only new development and
confirmation can address that limitation.

## Precision and decision

Under the declared normal planning model using overall parent-effect covariance,
the H2 gate's simulated joint superiority power at a true 0.5pp effect is about
11.2% for48parents and14.9% for64parents. Conservative joint-80% MDE estimates
are1.69pp and1.46pp, respectively. These projections are conditional on tiny,
overlapping CV fits and differ from the within-stream bootstrap assumptions;
they are planning warnings, not definitive confirmation sample sizes.

Shared-item uncertainty is material. Increasing item counts may improve scoring
precision, but cannot eliminate harvest/parent variability. Benchmark disjoint
item availability and inference cost before choosing fresh sample sizes. Do not
automatically spend the560-SU confirmation allowance on an underpowered design.

The machine decision is `REVIEW_B_development_signal_only_not_confirmation`.
The scientific interpretation is: admission has some operational headroom, the
tested learned rule is not yet practically superior, and a fresh balanced
16-parent development pilot is eligible for preparation and cost review. That
pilot must freeze rendering from harvesting through evaluation, retain independent
scoring, compare H1/H2/replay gates, and stop before C if headroom, discrimination
or power remains inadequate. Pooling is omitted under the current interaction
result. No Stage B or C GPU job is queued at this review.

## Accounting

Both completed CPU jobs incur the declared15-minute minimum: **0.165 SUs total**.
The persistent additional-work ledger now shows **87.79611 SUs charged plus
reserved** against1,200SUs; unused reservation from this chain has been released.
The account display remains1,997SUs available, a different accounting scope.
