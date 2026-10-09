# E1 fixed comparator and detectability controls

Declared before fresh balanced replication outcomes. This adds no architecture
search and does not change the A2 labeling or Gate 2 decision. Fit only on the
immutable old training/development partitions; the previously exposed test is
not a new confirmation population.

Use one fixed LightGBM adaptation for each physical immediate, future and
continuation target, with three independent target/OOD/retention regressors.
Inputs are exactly the existing pre-commit encoder allowlist: parent state,
candidate descriptors and availability masks, and previous accepted history
with masks. No observed candidate gains, labels, IDs, checkpoint paths or
post-update runtime fields enter inputs. No outcome-based feature screening.

Fixed settings: learning rate .05, four leaves, at most 140 trees, row and column
fractions .6, row sampling every iteration, minimum 20 child samples, L2 penalty
1, seed42, four threads, deterministic column-wise fitting. Stop after fifty
iterations without improvement in development mean squared error for each
head. Preserve every target, including unsuccessful fits, and verify exact
saved-model reload. This is a **TuneAhead-style cached-feature adaptation**, not
a faithful reproduction: our original probe has ten steps; the reference uses
one hundred. The target and recursive experimental setting also differ.

Reference: [TuneAhead](https://arxiv.org/abs/2606.17660). Published settings
motivate the bounded small-tree comparator; unavailable reference features,
their selected feature subset and additional fine-tuning runs are not recreated.
The fitting seed and row-sampling frequency above are fixed adaptation choices.

Also run the existing independent GRU fitting procedure on nine private
synthetic-label clones: seeds42/43/44 and injected noise levels0/.5/1. The known
signal is .01 times within-parent centered pre-update loss, standardized using
training-only mean/scale. Preserve actual pre-commit inputs and physical H1
auxiliary values. Never overwrite original H2 labels. Report all controls as
synthetic; success demonstrates detectability of this constructed signal, not
real forecasting success, statistical power or inherent unpredictability.

Dependency installation and all fitting happen on a scheduled CPU node in an
isolated venv that shares original packages. Pin LightGBM4.6.0, scikit-learn1.7.2,
SciPy1.16.2, joblib1.5.2 and threadpoolctl3.6.0, install only binary distributions
without dependency upgrades. The original model-training environment remains
unchanged. If installation fails, retain logs and correct only deployment.

The separate preliminary reliability review summarizes exactly the four
generation-zero noise parents, one per stream. It cannot replace the complete
twelve-state analysis and cannot release balanced replication automatically.
