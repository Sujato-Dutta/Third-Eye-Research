# Stage A execution declaration

Authorized by the October 18 research plan and the subsequent instruction to
start execution. This is an exploratory analysis of the already inspected,
corrected eight-parent dataset, not new confirmation. The original crossed
primary results and all source labels remain immutable.

## Inputs and admissible reuse

Use only the completed corrected campaign `crossed_corrections_a274ff920ac6`:
24 source-bound receipts, 216 matched-context evaluation artifacts, fresh parent
and M1 controls, and actual parent/candidate LoRA weights. Bind the historical
plan/report, every receipt and every consumed artifact to file hashes. Verify
aligned item identities, binary verdicts, context, checkpoint and donor identities.

The new authorized research plan explicitly requires exploratory admission fits.
Native corrected diagonal outcomes are therefore used in a newly declared,
development-only view. Preserve their original training-exclusion flags and
never feed counterfactual off-diagonal labels into those fits. This exception
does not authorize updating old forecasters, marking Gate 2 passed, changing
source outcomes or calling this dataset untouched. Off-diagonals remain used
only for interaction diagnostics.

Historical text/gradient probe features lack a verified matched rendering context.
Exclude them uniformly across streams for Stage A. Use corrected selection-half
H1 gains and the actual scaled-BA candidate-minus-parent Frobenius norm. Compute
the latter using compact Gram products on CPU: no base-model load, SVD, adapter
factor proxy or short-probe norm. Richer fresh probes remain a Stage B task.

## Fixed analyses

- Reuse the existing hash-based item split namespace. Select candidates on one
  half and score the other; average both orientations within each parent.
- Admission proxy: for the fixed immediate candidate, choose admit/hold from
  two continuation draws and the selection half, score on the third draw/other
  half. Average all three held-out-draw rotations. Zero gain holds. Uniform
  candidate ties are averaged as randomized decisions, not chosen favorably.
- Keep signed proxy utility. Never replace it with a noisy positive part.
  Compare it with forced updates, holding and the better population-constant
  action. The cross-fitted proxy is not the ideal oracle or an upper bound.
- Diagnose recipient/donor residual agreement across draws/item halves. Report
  observed candidate/draw/item variation without asserting separately identified
  harvest/optimizer components or additive causal variance shares.
- Selection optimism: all K=1/2/3 subsets, averaged within parents. Separate
  same-draw item selection effects from cross-draw selection gaps. Fixed nested
  item fractions are 0.25/0.5/1.0 of each half; no favorable fraction selection.
- Two linear gates: immediate-only or future-label training. Fixed ridge grid
  0.1/1/10/100, threshold grid 0/0.0025/0.005/0.01, predicted retention floor
  -0.005. Leave one parent out, validation index `(test+2)%8`, others train.
  Training and validation targets use only the selection item half; held-out
  scoring items do not train/calibrate any model, including other parents.
- Select ridge alpha by validation MSE; ties prefer stronger regularization.
  Select thresholds by that horizon's validation utility; ties prefer higher
  threshold. Retain folds/models/decisions and fixed-threshold curves.
- Replay baseline admits the fixed immediate candidate only when observed
  target gain is positive and OOD/retention gains are nonnegative.
- Uncertainty: 5,000 parent resamples, stratified two parents per four streams.
  CV fits overlap, so intervals are descriptive and conditional on the fits and
  finite item pool. Eight parents remain eight independent units.
- Separately resample scoring items 1,000 times with fixed decisions/models.
  Parents/backbones sharing identical role/item pools use the same item resample.
  This conditional item uncertainty is not added to or subtracted from the
  observed parent variance as an identified additive component.
- Planning: two-contrast normal simulations at N=32/48/64/80/96, effects
  0.5/1/2 utility pp, SD multipliers 0.75/1/1.5/2. Simultaneous two-sided intervals
  use Bonferroni correction. These are uncertain planning sensitivities, not
  confirmation power or the probability of meeting every success criterion.
- Include known synthetic equal-value and recoverable admission/selection
  controls. They do not establish real-task power.
- Include a sourced four-study descriptive precision audit. Missing paired
  variances/MDEs remain missing; no inference that our noise invalidates theirs.

## Stop and resource accounting

First run the complete current suite/lint in a Slurm CPU job; analysis depends
on its success. Validation reserves 20 minutes and analysis at most one hour
on the gg partition: combined maximum 0.44 SUs at 0.33 SU/node-hour. Both count
toward Stage A's 8-SU ceiling and the persistent 1,200-SU ledger. No GPU requested.

If the proxy-versus-better-constant upper interval is below 0.5 pp, report no
demonstrated operational headroom and stop before C. If its mean reaches 0.5 pp
and a gate has positive point gain versus both constants, request a Stage B
development review. Otherwise report uncertainty and a bounded precision option.
None is an automatic confirmation release or a universal impossibility result.
Do not select the better gate on these tiny folds as a publishable success.

All real artifact loading, norm calculation, fitting and statistics execute on
allocated compute nodes. Login handles staging, file hashes and scheduler metadata
only. Reports, failures, source and allocation receipts are retained. Analysis
ends at the scientific review; no pilot/confirmation is submitted automatically.

## Submission record

Immutable source:
`c8c37ae448f5531f146807c2e007f3a842e18b00de6a3ca26ef5d84fbc4b418e`.
Campaign:
`/scratch/11617/sujato_ts/third_eye_research/runs/admission_stage_a_c8c37ae448f5`.

Full validation **1078453** and CPU analysis **1078454** were accepted. Analysis
requires `afterok:1078453`, with invalid-dependency cancellation enabled. The
October 18, 11:03:20 IST check showed validation pending for Priority and analysis
pending on its dependency. No scientific Stage A outcome is available at this
check. Local validation passed nine targeted tests and complete source/test lint.

The persistent ledger now holds **88.07111 SUs charged plus reserved**, including
the new 0.44-SU chain. These are reservations until accounting completes. Failed
jobs and subsequent fixes still count toward the same caps; no retry is automatic.
This status append does not change the pre-execution declaration packaged with
the immutable source.

At12:12IST on October18, both jobs were still pending, with no allocated node,
failure log or scientific output. The gg partition showed230allocated nodes and
21reserved nodes. Slurm's tentative validation start was18:24:11IST that day;
this is a scheduler estimate, not a promised start or completion.

For backfill, the still-pending validation request was shortened from20to10minutes
and analysis from60to30minutes. Both updates were accepted. Source, test suite,
analyses, dependencies and resource accounting limits remain unchanged. The full
original0.44-SU reservation is conservatively retained. The scheduler adjustment
is recorded remotely in `backfill_adjustment_20261018.json`.

Both jobs subsequently completed successfully. Validation passed171tests and
lint; analysis finished in30seconds without a failure artifact. Completed cost
is0.165SUs, and the reconciled ledger totals87.79611SUs charged plus reserved.
The [results review](admission_results_20261018.md) records corrected harm,
independent admission headroom, shared-item uncertainty and the Stage B review
decision. No fresh pilot or confirmation was submitted automatically.
