# Forecaster extension F2: candidate comparisons and continuation residuals

Declared October 8, 2026, before new F2 empirical fits, following authorization
to develop a forecasting extension. This revises the forecaster, while frozen
A2 labeling, candidate construction, LoRA updates and evaluation rules remain
unchanged. Original fits, negative results and source remain archived intact.

## Mechanism and research claim

Third Eye-Contrast compares the complete K=3 set from a shared parent. It
predicts three immediate consequences and three standardized-continuation
increments, then adds them to forecast H=2 target/OOD/retention changes. A
shared component predicts the average response of the current candidate set;
zero-sum candidate components predict differences around that average. The
continuation latent receives the predicted immediate consequence vector,
candidate features and history context, never the actual post-update scores.

For candidate k, the model represents

    predicted_delta_1[k] = shared_1 + relative_1[k]
    predicted_delta_2[k] = predicted_delta_1[k] + shared_c + relative_c[k]

with each relative component summing to zero over the set. Its set operations
are permutation-equivariant: reordering candidates reorders predictions.
History remains the original two-layer GRU, with at most three prior states.
The full model uses hidden size sixteen and remains below one million
parameters. The physical aggregation weights remain the frozen equal weights;
no head is dropped or reweighted to improve results.

Set conditioning and state/relative-value decompositions have established
precedents: [Deep Sets](https://arxiv.org/abs/1703.06114) and
[dueling networks](https://proceedings.mlr.press/v48/wangf16.pdf). This extension
proposes applying a candidate-comparison decomposition to persistent RSI
updates and supervising the future continuation increment with paired cached
H=1/H=2 outcomes. It is not a claim that these general building blocks are new,
a causal identification claim, or evidence of a broadly capable world model.
The experimental contribution requires quality gains and supporting ablations.

## Fixed fifteen-fit comparison

- Full F2: H=1 and H=2 for seeds 42, 43, 44, six fits. Architecture and parameter
  count match between horizons. H=1 predicts immediate outcomes only; its
  continuation residual target is zero, and H=2 labels do not affect its fit or
  checkpoint selection. H=1 future-H=2 metrics are reporting only.
- Independent control: the original GRU with hidden size sixty-four, fitted
  using the new primary comparative objective and checkpoint criterion. Both
  horizons and all three seeds give six fits. This controls for changing the
  fitting procedure without adding joint/temporal supervision.
- Full-model H=2 ablations at seed 42: remove set coupling; remove immediate
  and continuation auxiliary losses; replace comparative loss with absolute
  vector regression and remove the pairwise ranking term. Three fits.

The primary method is **full H=2 seed 42**, fixed before results. A passing
alternative seed or ablation cannot silently replace a failing primary method.
Report all predefined results, actual parameter counts and runtime.

## Data and training rules

Use the immutable audited snapshot under
`runs/a2/provisional_forecasting_fast_20261008`, with its existing thirty-six
trajectory identities and split seed 42: 97 training states, forty validation
states and forty test states. Only train and validation are passed to F2
fitting. Test predictions/metrics are not computed by this development study.
Each model receives complete K=3 states and the same input allowlist. Identity,
runtime, continuation availability and actual future outcomes are excluded
from inference. Shared-parent context must agree within a set.

Feature normalization uses training values only. Consequence scales and the
ranking temperature use train-only H=1 within-state variation, shared between
the horizon arms. Scale floors are the frozen development evaluation
resolutions: 1/64 for target and OOD, 1/256 for retention; utility temperature
is at least 0.001. Outputs are restored to accuracy-fraction units before
aggregation and the existing metrics.

The main objective combines normalized within-state vector error (weight one),
candidate-set mean error (0.1), immediate and continuation auxiliary vector
losses (0.25 each), and a non-tied pairwise utility ranking loss (0.25).
Auxiliary vector losses use the same relative/mean decomposition. The H=1
arm uses H=1 primary/auxiliary supervision and zero continuation residuals.
The independent control has no temporal auxiliary heads. Ablation changes
are precisely those listed above.

All fits use AdamW, learning rate 0.001, weight decay 0.0001, gradient norm cap
one, whole-state batches of sixteen, maximum 200 epochs and patience forty.
Checkpoint selection uses own-horizon validation only:

    0.5 * (Spearman + informative_top1) - 0.01 * utility_RMSE

H=1 models are selected on H=1 validation performance. Undefined Spearman/top1
contribute -1/0 to this score. The criterion is fixed before this comparison,
and it contains no test metric or search over gate thresholds. This is one
declared architectural/fitting study, not an open-ended search for a passing
validation result. Validation has already been used for development; untouched
test and online-policy evidence are still required for final claims.

## Evidence and execution

Frozen Gate 2 is unchanged: H=2 Direct-role forecasts, at least thirty
nonconstant validation states, mean per-state Spearman >=0.30, informative
top-1 >=0.45 and above tie-adjusted chance. Report head errors/signs/ranks,
matched H=1 future-H=2 rankings, and all ablations. Passing this gate does not
prove that F2 beats the controls or improves recursive trajectories.

All scientific fitting and full validation run in Slurm compute allocations.
The thirty-minute limit bounds the CPU job. Original A2 source and operational
hashes are checked before every fit; extension source, launchers, this document,
snapshot manifest and model artifacts receive separate hashes. Model reload
must reproduce validation predictions exactly. The loader records an explicit
extension and architecture; `direct`/`matched_h1` identify online selector
roles, not a claim that new weights are the original GRU.

The extension provides a compatible online entry point with additional source
provenance. This study stops at review and submits no online/scaling jobs.
Correction attempts, scanned training portions, unique candidate batches,
K=3, T<=5, H=2 continuation, optimizer budgets, terminal semantics, manifests
and final-test access rules remain frozen.
