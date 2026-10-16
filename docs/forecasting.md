# Third Eye forecasting implementation

The core implementation now supports `--kind expected` and online
`--policy expected`. This is an in-place improvement of Third Eye, with the
existing forecasters retained as reproducible baselines. Historical experiments
and their immutable source snapshots remain unchanged.

The October 11 evidence motivates this objective: fresh continuations often
change candidate rankings, the frozen single-draw H2 predictor did not beat its
H1 comparator on fresh trajectories, and forced updates can lose value relative
to keeping the parent. These results guide development; they do not demonstrate
an advantage for this revised implementation.

## Prediction and decision

The small MLP consumes the existing precommit allowlist, the last three history
entries, and candidate features centered within the current K=3 set. Missing
features have availability bits and are excluded from sibling means. IDs,
future correction availability, H1/H2 labels, continuation batches and future
metrics never enter the encoder. Existing diversity, interference, reversible
probe, drift and correction-supply features remain available.

The model predicts three immediate consequences and three continuation gains:

`expected_H2 = predicted_H1 + predicted_continuation_gain`.

A joint positive-definite covariance head models continuation variability across
target, OOD and retention. Utility standard deviation is `sqrt(w' Sigma w)`;
cross-head covariance is included. Supervision uses the unbiased covariance of
repeated draws centered at their empirical mean, rather than folding mean-fit
error into the variability target. A pairwise loss emphasizes within-parent
utility differences supported by the paired continuation draws.

This covariance describes continuation draw variability conditional on M1 and
the declared evaluation items. It is **not** epistemic uncertainty, uncertainty
in the estimated expected value, or a calibrated confidence interval. Three
draws are a collection minimum, not an established universally sufficient count.
The audit reports matched-seed winner agreement and within-versus-between
variation before anyone treats the target as reliable.

The default decision selects predicted expected utility. Optional
`--risk-penalty LAMBDA` selects mean minus lambda times draw standard deviation.
With `--allow-abstain`, the parent is retained if every risk score is <=0. These
settings are saved before evaluation; they are not tuned on test outcomes.
Abstention performs no full candidate training and records its actual zero-step
action. Probes and harvesting still have costs. Holding the parent has utility
zero in this reference; this does not establish that future online policy
trajectories improve. Ties are reproducible uniform draws for online actions and
tie-averaged in diagnostic comparisons.

## Repeated labels

Run label collection **inside a Slurm GPU allocation** using a new output:

```bash
python experiments/run.py --config CONFIG --manifest MANIFEST --output NEW_RUN \
  --mode labels --policy random --continuation-seed-offsets 50000000 100000000
```

This collects the original continuation plus two fresh continuations per
candidate. Offsets are recorded before collection and matched across K=3. Each
fresh draw reloads the exact saved M1, harvests under the declared protocol and
executes the same continuation budget. Candidate batch size, LR, LoRA and
optimizer budget are unchanged. Correction-starved draws preserve M2=M1 and
execute zero optimizer steps. Only M1 remains eligible for trajectory commitment.
Failure leaves diagnostic files and publishes no partial K=3 state.

Native `labels.h1` and `labels.h2` retain their original meanings. The additional
`continuation_replicates` field binds each draw's seed, starting M1 hash, ending
hash, evaluation manifest, consequence deltas, availability and optimizer budget.
Single-draw labels are rejected for expected-value fitting; they are not cloned
or assigned zero variance. Mean targets are empirical estimates, not noiseless
population expectations.

## Fit and independently score

Fit on a compute node, with whole trajectories assigned before development:

```bash
python experiments/train_forecaster.py --labels NEW_RUN/meta_labels.jsonl \
  --output NEW_FORECASTER --kind expected --device cpu \
  --validation-fraction 0.15 --test-fraction 0.15
```

Multiple label paths are accepted. The 70/15/15 fractions apply to independent
trajectory components with rounding; historical defaults remain 60/20/20.
Normalization uses training only, early stopping uses validation only, and test
inputs are encoded after checkpoint selection. Save the model and decision
settings before collecting fresh scoring outcomes. CPU fitting is small enough
to use a Slurm CPU node; CPU execution on the login node is blocked by the CLI.

For a fresh evaluation, use held-out trajectories and a scoring manifest with
the **same harvesting train pool** and development/proxy evaluation items
disjoint from all original selection, feature-anchor and training items. Final
benchmark test sets remain sealed. Inside a GPU allocation:

```bash
python experiments/score_forecaster.py --config CONFIG \
  --selection-manifest SELECTION_MANIFEST --scoring-manifest SCORING_MANIFEST \
  --labels FRESH_HELD_OUT_RUN/meta_labels.jsonl --forecaster NEW_FORECASTER \
  --scoring-seed-offsets 150000000 200000000 250000000 \
  --verifier third_eye.evaluation.benchmarks:make_verifier --output NEW_SCORING
```

Code tasks require the existing isolated sandbox configuration. The scorer
freezes and writes the precommit decision before measuring any scoring outcomes.
It checks hashes, trajectory exclusions, item disjointness, utility weights and
fresh seeds, then independently scores the same K=3 M1 checkpoints. Nothing is
committed. The primary is mean-only forecast selection minus observed H1
selection on those independent outcomes. Predicted H1, random, no-update,
risk-sensitive selection and abstention effects are separate comparisons, with
whole-trajectory CIs and multiplicity correction for secondary tests. There is
no same-outcome noisy-oracle regret. CIs are conditional on the scoring item pool.

`run.py --mode online --policy expected --forecaster NEW_FORECASTER` deploys
the saved decision rule without future labels. Resume binds both saved weights
and complete forecaster metadata, including the abstention/risk settings.

## Existing evidence and deadline

`review_continuations.py` can audit and fit on the eight already completed
matched-control parents without new GPU collection. Its output is explicitly
exploratory: these outcomes have already been examined. It can check the
implementation and conditional target signal, but cannot restore the paper's
forecasting-superiority claim or turn those parents into a fresh test set.

The 40-state confirmation has also been inspected and must not be repurposed as
an untouched confirmation for the revised model. A credible advantage still
requires fresh, sufficiently powered repeated-continuation trajectories and
independent scoring. Collecting 150–300 new parents is not a credible promise
for the remaining October 11 window. The immediate deadline work is bounded
validation and an existing-data signal audit, then an evidence-based decision.

`experiments/jobs/forecast_validation.slurm` runs the full implementation suite
on a compute node. `continuation_review.slurm` consumes a matching successful
validation receipt before the bounded existing-data audit and exploratory fit.
Execution snapshots identify the current main source hash while preserving
source snapshots used by pending historical report jobs. They are provenance
copies, not separate forecasting implementations or scientific branches.

## Execution status: October 11, 09:34 IST

Local implementation verification completed: **131 tests passed**, including a
synthetic positive control; lint and format checks passed. Synthetic outcomes
are not research evidence. The two subsequently added CLI allocation guards
passed static checks. Compute-node validation of the staged source is pending.

Current main source SHA:
`7fe4032b6b99fbe262ecd0c7129742bf819a59e32c972a7ae1f5ed63b76f8d6b`.

| Vista job | Work | Last checked status |
| --- | --- | --- |
| 1064655 | Compute-node full suite and source verification | PENDING: Priority |
| 1064656 | Eight existing matched parents: signal audit and exploratory fit | PENDING: afterok1064655 |

Both jobs use the main source execution snapshot
`/scratch/11617/sujato_ts/third_eye_research/releases/main_7fe4032b6b99` and
write to `/scratch/11617/sujato_ts/third_eye_research/runs/expected_7fe4032b6b99`.
The scheduler owns the dependency; it does not need a local editor or SSH session
to remain open. No fresh GPU collection or confirmatory superiority result has
been submitted here. The older report repair remains independent:
1064590 completed successfully; 1064591 was still pending at this check.

## Execution update: October 11, 21:58 IST

1064655 failed during test collection because the staged package omitted
pytest.ini and pytest inherited the original project's config, which inserted
the historical `src` before the revised implementation. 1064656 was cancelled
by its failed dependency. Both attempts and their source snapshot are preserved.
The packaging now includes pytest.ini, and validation explicitly sets the
snapshot-local config and root. No model or scientific parameter changed.

Replacement main source SHA:
`5bd72d22a15ccab5b680dc30fe3fd36b903632dc9301abdb40fbd7dfb06980f0`.

| Vista job | Work | Last checked status |
| --- | --- | --- |
| 1065378 | Full suite on the correct main source | RUNNING on i614-081 |
| 1065379 | Existing-parent signal audit and exploratory fit | PENDING: afterok1065378 |

The replacement execution snapshot is
`/scratch/11617/sujato_ts/third_eye_research/releases/main_5bd72d22a15c`,
with outputs under
`/scratch/11617/sujato_ts/third_eye_research/runs/expected_5bd72d22a15c`.
Historical report/archive1064591 has completed. Improved-model performance is
still pending and this existing-data fit remains exploratory.

## Importer repair: October 11 evening

Full-suite validation1065378 passed on Vista:131 tests and lint. Audit1065379
then stopped in12 seconds on a strict native-versus-reevaluated H1 comparison.
Some matched control scores already recorded in the final report differ from
the original scores despite identical checkpoint hashes. The importer now
preserves both: native `labels.h1/h2` remain unchanged, while the optional
hash/manifest-bound `continuation_reference` records matched re-evaluated H1
supervision and native-versus-control differences. Expected-value fitting and
terminal consistency use that reference when present. It is excluded from
precommit inputs. New native collection still uses its own original H1 normally.

Twenty-one targeted tests passed, including reference isolation and preservation
of historical labels. Full validation1065386 and dependent audit/fit1065387 have
been submitted with source SHA
`b1cc94beb828cb716b427cc44fecc0f45a30ea6b57d5ca227c8f9ce1dad40309`.
Their source/output directories end in `main_b1cc94beb828` and
`runs/expected_b1cc94beb828`, respectively. Earlier failures remain intact. This
is an importer/supervision consistency repair, not a changed continuation budget
or a claim of new forecasting success.

## Completed audit/fit: October 11, 22:08 IST

1065386 completed successfully in55 seconds: **132 tests passed** and lint
passed on Vista. 1065387 completed successfully in21 seconds, producing the
eight-parent audit, repeated-label dataset and expected-value forecaster.
Both ran on allocated compute nodes; the historical source snapshots and
outcomes remain intact.

The exploratory split has four training parents, two validation parents and
two test parents, each with K=3 and three continuation draws. On the two test
parents, top-1 is50% versus33.3% chance and Spearman is0.25. This is one correct
winner out of two previously inspected states. It is not fresh confirmation,
not a Gate-2 pass, and not evidence of a reliable selection-value advantage.
The implementation and fitting pipeline now complete successfully; a forecasting
superiority claim still requires adequately sized fresh independent scoring.

## Fresh-data pilots after the submission-date change

The submission target moved to late December. The next bounded campaign is
specified in [the December research plan](december_research_plan.md): eight fresh
development states, K=3 and five continuation outcomes per M1, with A2 training
and harvesting unchanged. `experiments/forecasting_pilots.py` checks provenance,
validation receipts and complete per-item evaluations before reporting structural
success. Scientific execution requires Slurm allocations. Larger collection is
not automatically released by this pilot stage.

`experiments/run.py --capture-item-outcomes` retains individual verifier results
from the original scoring pass. Terminal outcomes refer to the existing M1
evaluation, preserving zero executed continuation steps. Resume checks retain
the same continuation seed grid and capture setting. Independent scoring also
captures its own disjoint items, rather than reusing selection outcomes.

## Forecasting review after successful pilot inspection

All eight pilots completed and structural inspection passed on October13.
The [bounded forecasting review](forecasting_review_20261013.md) scores the
checkpoint frozen before pilot collection and runs leave-one-trajectory-out
development comparisons with H1, mean-H2 and linear models. It also separates
draw selection from draw scoring. `experiments/review_forecasting.py` requires
an allocated compute node and a hash-pinned frozen checkpoint. It rejects
evaluation-state overlap with that checkpoint's training/model-selection data.
Eight inspected initial-state pilots remain insufficient for final forecasting
confirmation; the review does not automatically pass Gate2 or expand collection.

The October13 review completed successfully after139 tests and lint passed.
Both the old frozen expected checkpoint and the leave-one-trajectory-out
expected ensemble achieved12.5% top1 versus37.5% tie-adjusted chance. Matched
H1 reached50%, mean-H2 Direct62.5%, and linear H250% withrho0.438. These are
eight-state development findings; the expected architecture has no demonstrated
advantage, and a useful future-versus-immediate selection advantage remains
unproven. The [full review](forecasting_review_20261013.md) records contrasts,
limitations, provenance and the bounded next-step recommendation.

The [October13 additions plan](bounded_additions_20261013.md) caps subsequent
work at1,200 additional SUs, targeting about1,000. The main online runner now
supports `guarded_h1`: select from positive-immediate-utility candidates whose
measured retention does not decrease, otherwise preserve the parent. It evaluates
K=3 immediate branches and commits the saved winner without another optimizer
update. This is a fixed experimental rule, not a validated safety guarantee.
The first scheduled comparison halves candidate LR on fixed verified batches,
retains baseline continuation settings, and stops for scientific review before
balanced collection or recursive confirmation is released.
