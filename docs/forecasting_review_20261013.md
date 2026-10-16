# Bounded forecasting review: declaration before execution

October13,2026. The eight fresh-data pilots and structural inspection have
passed. This next stage evaluates forecasting using those saved outcomes, with
no new model-update experiments or GPU jobs. The collection and A2 protocol are
unchanged. The stage remains development evidence and cannot release a larger
cohort or mark Gate2 passed automatically.

## Fixed comparison

The primary comparison is selection utility from the existing expected-value
checkpoint versus observed immediate-gain selection on the eight pilot states.
The checkpoint was trained and saved before pilot collection; its weights SHA is
`b0e4ec855be88adc6a98f224df70ce12cf3c79d16af786074cd6c8607d68c2e4`.
Its source SHA is
`b1cc94beb828cb716b427cc44fecc0f45a30ea6b57d5ca227c8f9ce1dad40309`.
Evaluation state/trajectory overlap with its training or model-selection data
is rejected. Weights, normalization, decision rules and ensemble choices are
not tuned on pilot outcomes. The evaluation design is retrospective, even
though the checkpoint predates data collection; this is not a preregistered
independent-item confirmation.

Utility remains the frozen equal-weight average of target, OOD and retention
accuracy changes. Candidate predictions use only precommit fields. Tied
selectors receive uniform candidate weighting, rather than a favorable
candidate-order choice. Forced selection is primary. Random and no-update
are explicit comparators; abstention is a separate diagnostic.

## Development comparisons

Leave one whole trajectory out in each of eight folds; each trajectory contains
one initial K=3 state. Read tasks in their frozen index order. Foldi uses
trajectoryi as test, trajectory(i+2)mod8 as validation and the remaining six as
training. Each state is predicted once; folds never split a trajectory.

Compare expected-value, the existing matched_h1 temporal model and Direct H2.
Use hidden16,100 epochs maximum, patience15, LR1e-3 and fixed model seeds
42/1042/2042. Average all three seeds' predictions uniformly; never select a
seed using test results. Direct uses a temporary mean-of-five H2 target view;
historical native labels are preserved. Expected-value retains its existing
mean, covariance and ranking objectives without architecture changes.

Linear H1/H2 models use the same full candidate-relative feature allowlist;
a simple H2 model uses baseline scores, NLL, candidate-probe loss change,
retention-probe loss change and retention-gradient cosine, with availability
masks and candidate-relative values. Normalize from training only. Choose
ridge alpha from0.1/1/10/100 using validation MSE only. The expected model's
H1 component is reported separately; it shares H2-trained representations
and is not an independently trained H1-only control. The existing matched_h1
architecture matches Direct, not the newer expected-value MLP.

For each state, select a privileged future candidate using three continuation
draws and score it on the other two, averaging all ten disjoint partitions.
This prevents selecting and scoring an observed best candidate on the same
draws. It measures whether continuation selection has value, not whether a
precommit predictor can recover it. All draws share finite evaluation items;
this diagnostic does not replace disjoint-item scoring.

Report tie-aware top1, chance, mean state Spearman and paired utility contrasts
against H1, linear models, random and no-update. Bootstrap/permutation units
are whole pilot trajectories, not candidate rows, model seeds or draw rotations.
Cross-validation training sets overlap, so its intervals are descriptive and
conditional on fitted predictions. Apply Holm adjustment to secondary
contrasts; the frozen-checkpoint-versus-observed-H1 contrast is primary.
No risk coefficient is tuned: beta1 and a zero no-update threshold are fixed
development diagnostics. Risk uses conditional draw spread, not a calibrated
confidence interval or epistemic uncertainty.

## Execution and decision boundaries

New main-source code gets full CPU validation before the review runs. Both
jobs use Slurmgg compute allocations; login performs staging/submission only.
Validation has20 minutes maximum and review45 minutes maximum: together under
0.4 SUs at the previously verified0.33/node-hourgg rate. No fresh GPU allocation
is requested. Outputs and failures are preserved in a new execution directory.

A positive preliminary comparison justifies power/precision planning and a
larger frozen cohort. A negative or uncertain result requires examining signal
and sample-size limits; it is not a universal forecasting impossibility claim.
Neither outcome is allowed to become a final Gate2 pass, a superiority claim
or automatic broad scaling from these eight initial-state development pilots.

## Submission record: October13,10:23 IST

Full CPU validation1068719 and dependent review1068720 were submitted.
The review hasafterok:1068719 andkill-on-invalid-dependency enabled. At the first
scheduler check validation was pending for Priority and review pending on its
dependency. Four targeted synthetic tests and lint passed locally before
submission. These test held-out trajectory coverage, test-outcome isolation
for ridge fitting, preservation of native labels, disjoint draw rotations and
the frozen-model overlap guard. The exact cluster snapshot is then checked by
the full suite before any real forecasting analysis.

Source SHA:
`6a4e1c71172b119d211cf292e6e28bb220617d32a5c66404c10320b975b617c1`.
Immutable source:
`/scratch/11617/sujato_ts/third_eye_research/releases/main_6a4e1c71172b`.
Outputs:
`/scratch/11617/sujato_ts/third_eye_research/runs/forecasting_review_6a4e1c71172b`.
Review outputs include the declaration, frozen predictions, each fold's saved
forecasters and split audits, out-of-fold predictions,report.json and a source
and input-hash-bound completion receipt. Existing pilots and checkpoint files
are read without modification. The main implementation remains on the current
project; the immutable execution snapshot is provenance, not a new branch.

## Completed results: October13,10:27 IST

Validation1068719 completed successfully in70s:139 tests passed in56.30s
and lint passed. Review1068720 completed successfully in59s. Both ran onCPU
compute nodes. At the published minimum charges the combined cost is0.165 SUs.
The eight input pilot states retain24 candidates and120 continuation outcomes.
All eight development folds completed. Native labels, the frozen checkpoint
and earlier source snapshots remain unchanged.

| Selector | Tie-aware top1 | Mean state Spearman |
|---|---:|---:|
| Checkpoint frozen before pilot collection | 12.5% | -0.483 |
| Expected-value, leave-one-trajectory-out ensemble | 12.5% | -0.329 |
| Existing matched H1 temporal baseline | 50.0% | 0.142 |
| Direct trained on mean-of-five H2 | 62.5% | 0.267 |
| Full linear H2 | 50.0% | 0.438 |
| Simple linear H2 | 50.0% | 0.313 |
| Observed immediate-gain selection | 62.5% | 0.483 |

The tie-adjusted random top1 reference is37.5%, rather than a blanket33.3%,
because one state has two tied mean-H2 winners. The frozen expected checkpoint
and the new cross-validated expected ensemble each choose one correct winner
out of eight states. They do not demonstrate a forecasting advantage.

The primary frozen-checkpoint minus observed-H1 selection contrast is
**-3.66 utility percentage points**, with a descriptive bootstrap interval
[-7.34,-0.45]pp and exact paired permutationp=0.125. The interval and test have
different small-sample behavior; neither justifies a population-level negative
forecasting claim. Expected ensemble minus matched H1 is-1.93pp; its secondary
Holm-adjustedp=0.8125. Observed immediate selection already attains the same
top1 as Direct. Better ranking numbers from another model therefore do not
establish a practically useful improvement over immediate selection.

Adding the learned continuation mean did not change any selected candidate
relative to each expected model's own H1 component in these eight states.
This is a diagnostic of the current fitted models, not evidence that future
consequences can never improve selection. The conditional covariance head and
its fixed risk rule also did not establish a gain over no-update. No-update
remains an essential comparator because many forced updates have negative
future utility under this protocol.

The simpler future predictors have more promising development rankings, but
they were evaluated among multiple predeclared comparators on only eight
initial-state trajectories. Full linear H2'srho0.438/top150% does not constitute
a Gate2 pass: the cohort is tiny, the review is exploratory, and shared finite
evaluation items remain a limitation. The draw-selection diagnostic is mixed
across streams and uses privileged future observations; its rotations must
not be treated as extra independent trajectories or as learned predictions.

The forecasting story is **not restored by this review**. The next justified
science step is a power/precision and learning-curve plan for a bounded,
balanced whole-trajectory comparison of simple mean-H2 predictors against
H1, with independent continuation seeds and scoring items. Do not spend the
remaining allocation scaling the current expected architecture on the basis
of these results, and do not declare an inherent forecasting limit from them.
Model choice and sample sizes for that larger stage must be frozen before its
confirmation outcomes are exposed.

The report, completion receipt, frozen predictions and submission receipt were
downloaded under`runs/deployment/evidence/forecasting_review_20261013`.
Report SHA256:
`80392773fa6a6d5b2353a04647f40a9658ce0ce0e3f9a77237f15a0458f5f2f9`.
The report isdevelopment_only=true,gate2_passed=null andautomatic_expansion=false.
