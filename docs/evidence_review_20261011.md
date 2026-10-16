# Completed evidence review: October 11, 2026

Subsequent work improves the core Third Eye forecaster to predict expected
continuation value and joint variability; see [the forecasting guide](forecasting.md).
This does not change the historical results or establish architectural
superiority. Initial validation1064655 failed during pytest collection because
an ancestor pytest configuration selected the historical source; dependent
audit1064656 was cancelled. Including pytest.ini in the main execution snapshot
and explicitly setting pytest's config/root fixes the deployment path without
changing the forecasting objective. Replacement validation1065378 is running,
with exploratory audit1065379 held by its afterok dependency as of21:58 IST.

Later update:1065378 passed all131 tests on Vista. Audit1065379 failed in12
seconds on a strict historical-versus-control score equality check. The final
report already records nonzero re-evaluation differences for some matched
checkpoints. The importer now preserves native labels and binds the matched
immediate supervision separately, without introducing it into model inputs.
Twenty-one targeted tests passed. Replacement validation1065386 and its
dependent exploratory audit1065387 are submitted; outcomes remain pending.

At22:08 IST both replacements completed successfully. Validation1065386 passed
132 tests on Vista. Audit/fit1065387 covered eight existing matched parents;
its two-parent exploratory test split gives top-150% and Spearman0.25.
That tiny, previously inspected split does not establish forecasting advantage
or supersede the original confirmation conclusions below.

All GPU collection has ended. The original cohort has 36 complete T5 trajectories
(180 states/540 labels); the balanced fresh cohort has eight complete T5
trajectories (40 states/120 labels). SmolLM contributes four separately declared
early states (12 labels), not a T5 replication. All eight matched-control parents
completed. Fresh confirmation and R1 analyses completed on Slurm CPU nodes.

The original core noise-report job1063239 failed on diagnostic list indexing
before writing its report/archive. Its log and accounting are preserved. A
separate CPU wrapper converts native utility lists to arrays before the unchanged
regret formula; three checks passed locally. Validation1064590 and dependent
report/archive1064591 completed successfully. This changes no data, training or
analysis definition. The native eight-parent report is complete, and the
durable core archive is17,672,665,724 bytes under
`/work/11617/sujato_ts/vista/third_eye_results/empirical_checkpoint_fallback_v3.tar.gz`.
Its SHA256 is
`927339ab105a0ba20b2d45c4e198aaafca127eab8cc3a824eeafd75ecd373ce2`.
The success/archive receipts and final native report have been retrieved locally.

## Primary results

Utility is the equally weighted mean of target/OOD/retention accuracy differences.
The table scales utility differences by100 into percentage points; it does not
describe target accuracy alone. Intervals resample trajectories within stream.

| Declared comparison | Estimate | 95% interval | Supported interpretation |
| --- | --- | --- | --- |
| Fresh frozen Direct-H2 minus matched-H1 future-selection value, 40 states/eight trajectories | approximately 0.00 pp | [-0.60, +0.63] pp | No demonstrated H2 advantage |
| R1 independent-seed, disjoint-item future selector minus immediate selector, eight parents | +0.34 pp | [-0.55, +1.24] pp | Independently scored selection advantage remains unresolved |
| R1 future selector minus uniform random, secondary comparator | +1.61 pp | [+0.88, +2.34] pp | Expensive observed-future selection retains some cross-seed value; this is not a learned forecast |
| R1 immediate-selected future utility relative to holding the parent | -1.96 pp | [-3.59, -0.33] pp | Forced candidate-plus-continuation updates can lose value under this protocol |
| R1 two selection continuations minus one, held-out continuation scoring | -0.21 pp | [-0.66, +0.23] pp | No demonstrated selection benefit from two repeats |

The fresh frozen Direct-H2 utility ranking has Spearman0.059 and top-1 30%,
versus tie-adjusted chance35%. The primary is the paired selection-value
comparison above, not post-hoc selection of a better-performing fit. Original
full-data refits remain separate from the frozen confirmation methods.

## Matched ranking variation

On the eight control parents, original H1/H2 strict-winner disagreement is62.5%.
Changing only the fixed-continuation optimizer seed yielded no observed winner
reversal. Fresh continuation variation averaged68.75% disagreement with original
H2. Original H1/H2 disagreement minus fresh-continuation variation is -6.25
percentage points, with interval [-37.5, +25]. Thus raw horizon disagreement is
not established as larger than the measured continuation variation. The fixed
optimizer and fresh-continuation controls represent different stochastic
processes; neither is a universal noise floor.

The independently scored future selector's secondary advantage over random
also argues against claiming that all future information is absent. Winner
changes, finite-sample uncertainty and failure of the tested forecasters do not
establish intrinsic unpredictability. This nominal interval is secondary and
does not turn the unresolved primary future-versus-immediate gap into a pass.

Observed-future-oracle regret on the 40-state cohort is1.34 pp [0.74,2.07], but
it chooses and scores the best measured future on the same outcomes. Do not
substitute this positive descriptive number for R1's independently scored gap.
Likewise, a confidence interval containing zero does not prove equivalence or
intrinsic unpredictability. Eight independent units and shared finite item pools
limit power and population generalization.

## Paper decision

The architectural-superiority headline is unsupported. The current evidence
also does not support a repeat-stable, independently evaluated lookahead
selection cost beyond continuation variability. Lead, if submitting this package,
with a controlled evaluation protocol and continuation sensitivity in persistent
parameter self-updates. Keep forecasting and observed regret as secondary,
carefully bounded findings. A provisional evidence-aligned title is
**Continuation-Sensitive Selection in Recursive Parameter Self-Updates**.

The practical takeaway is to separate selection from scoring across continuation
seeds and evaluation items, report continuation sensitivity, and include a
no-update reference rather than evaluating only forced updates. The data do not
establish a universally sufficient repeat count or a deployed abstention policy.
Mechanism associations remain correlational; one nominal diversity association
does not establish a causal explanation or multiplicity-adjusted discovery.

Claims remain limited to 3–4B models, LoRA, K=3, H=2, T=5 random-policy trajectories
for two backbones, and four early SmolLM sanity checks. No universal RSI ceiling,
long-horizon impossibility, full-parameter conclusion or Main/Oral acceptance
claim follows. Preserve the failed historical replay and reporting attempts.
