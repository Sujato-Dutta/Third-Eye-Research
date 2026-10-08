# Early A2 forecasting, October 8

Latest outcome: replacement **1057179** completed all eighteen fits and
reports in **91 seconds**, with exit 0:0, after **124 tests passed**. The
177-state snapshot passed its artifact audit. The fixed seed-42 H=2 Direct
result misses Gate 2: validation Spearman **0.215849**, top-1 **0.50**, forty
nonconstant states, chance **0.341667**. The correlation threshold remains
0.30; the accuracy threshold remains 0.45. No online/scaling release occurred.

All three Direct seeds miss the correlation threshold: 0.215849, 0.184151,
0.212500. Their validation top-1 values are 0.50, 0.475, 0.475, versus 0.35
for each matched-H=1 seed. These are descriptive results on the same validation
states, not independent replications or a significance claim. The fixed
primary seed remains 42; no favorable seed or ablation replaces it.

The audited A2 phenomenon screen passes on this snapshot: 53.67% strict
H=1/H=2 winner disagreement, 83.99% harmful-component candidate updates,
and zero terminal candidates. This establishes the phenomenon on the available
sample, not forecasting success or recursive policy improvement.

To avoid waiting for the three remaining GPU states before diagnosing the
forecaster, one matched schedule diagnostic is preregistered separately below.

CPU job **1057170** was submitted at 12:35:25 IST to forecast from existing
published labels while GPU recovery continues. It requests eight CPUs in `gg`,
four hours maximum, and no GPU. At 12:37 IST it was pending for Priority with
no dependency on label generation. Allocation time is not guaranteed. Its full
research suite and prefix checks must pass, followed by the label audit,
before any fit.

The metadata check contains 177 complete K=3 states / 531 labels across all
thirty-six original trajectory identities. The frozen split seed 42 gives:

| Partition | Published states | Planned maximum | Trajectories |
| --- | ---: | ---: | ---: |
| Train | 97 | 100 | 20 |
| Validation | 40 | 40 | 8 |
| Test | 40 | 40 | 8 |

All three missing states belong to training trajectories, indices 20, 22 and
24. No held-out state is missing. These are metadata counts; artifact audits
still precede fitting. Forty validation states do not establish that Gate 2's
minimum thirty nonconstant states are available; that is measured from labels.

The immutable snapshot is made at job execution before forecasting results.
Only ledger-published states with all three complete labels enter it. No
trajectory is dropped, replaced or split across partitions. Record hashes,
actual sample counts and trajectory assignments are saved in its manifest;
original data and state artifacts remain unchanged.

All eighteen fixed fits share this snapshot and retain the frozen settings,
seeds, features, utility weights and split. Two independent four-thread CPU
processes run concurrently. The matched H=1/H=2 comparisons, Direct/Dynamics
and six ablations remain included. Original phenomenon/Gate 2 calculations
apply; no harvesting, training-update, terminal or gate rule changes.

Results are isolated in `runs/a2/provisional_forecasting_20261008` and stop for
review. No canonical model or gate artifact is replaced, and no online,
transfer or scaling job launches automatically. The report can assess
forecasting on the audited available sample before recovery finishes.

The full-data CPU audit/fit chain remains queued as later confirmation. It
reuses completed GPU work. Claims for models trained on the full recovered
dataset require their own fits/evaluations; early results must identify their
actual sample. Review must verify that trajectory assignments agree between
the snapshot and full dataset. Missing states remain recorded and preserved.

Evidence: `runs/deployment/evidence/a2_prefix_split_readiness_20261008.json`,
`a2_prefix_forecast_release_20261008.json`, and requests 348/349 in
`runs/deployment/vista_monitor_requests/`. Supplemental operational files are
hashed separately; frozen A2 scientific source and original tools stay intact.

## Deadline-oriented scheduling replacement

Job 1057170's live start estimate was around 18:12 IST. Before it started,
1057179 was queued in both compatible `gg` and `gh-dev` partitions with
seventy-two requested CPU cores and a two-hour limit. Vista's allocation
check accepted the request; only the old pending job was canceled. Slurm
allocated the replacement in `gg` at 12:46:24 IST, and it completed at
12:47:55 IST. The first two fits were the fixed seed-42 Direct and matched H=1;
their review report appeared before the remaining sixteen fixed fits, which
ran in independent four-thread processes. Gates and fit settings stayed intact.

## Single forecaster stopping diagnostic

Job **1057195** was queued at 12:59 IST, requesting a fifteen-minute compute
allocation in `gg,gh-dev`. It is a diagnostic, not a replacement result.
Direct seed 42 selected epoch 3 and stopped at epoch 23 under patience 20.
Training-side correlation is also weak; this observation suggests checking
optimization but does not establish that stopping is the cause.

The single predeclared change is patience **20 to 200**, keeping the existing
maximum **200 epochs** and the original validation checkpoint-selection score.
Both matched H=1 and H=2 use this schedule for seeds 42/43/44, with primary
decision seed still 42. The immutable 97/40/40 snapshot, architectures,
features, normalization, optimizer settings, utility weights and split stay
unchanged. Test metrics are not used for selection. Candidate construction,
LoRA budgets, harvesting, terminal semantics and science thresholds stay frozen.

Two new operational checks precede the six diagnostic fits. Results go into
`runs/a2/forecaster_patience_diagnostic_20261008` and stop for review. Original
failed gate results and weights remain intact. There is no automatic adoption,
online launch or scaling, and no open-ended hyperparameter search.

The diagnostic completed successfully in **48 seconds** after both operational
checks passed. All six fits ran 200 epochs. Each Direct seed retained the same
validation correlation/accuracy as its original fit; primary seed 42 remains
0.215849 / 0.50. Extending patience did not resolve the gate miss, so this
diagnostic is not adopted and no passing result is claimed. This rules out
that single stopping change as a remedy under the existing checkpoint selector;
it does not identify a cause or prove that future-value prediction is impossible.

At 13:02 IST, recovery validation 1057163 completed after **129 tests passed**
and submitted GPU recovery **1057197**, which was pending. Original task 22
also timed out; task 24 was still running. The original array now has 33 complete
tasks, two timeouts and one running task. Recovery remains separate from the
completed forecasting review and cannot release scaling.

Evidence: `runs/deployment/evidence/a2_prefix_forecast_review_20261008.json`,
`a2_prefix_forecast_fast_evidence_20261008.tar.gz` (SHA-256
`d45b097fe8f30a33141e9a87d6035c4453256ad696d6bcfe4d0e45ea2daceef2`),
`a2_patience_diagnostic_release_20261008.json`, and queue receipts 356/362.
Diagnostic results: `a2_forecaster_patience_review_20261008.json`; completion
and recovery validation evidence: request responses 366/367.
