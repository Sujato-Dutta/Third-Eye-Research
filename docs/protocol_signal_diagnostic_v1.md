# Signal diagnostic v1: predeclared development study

Frozen October 8 before diagnostic fitting, following authorization for a
thirty-minute direction assessment. This changes no A2 labels, training updates,
gate, F2 architecture, or final experimental scope. No GPU harvesting, online
policy or scaling launch is part of this study.

Use the immutable audited F2 input snapshot: 97 training states in 20
trajectories and 40 development states in 8 trajectories. Exclude the test
partition from diagnostic fitting, analysis and model selection. Preserve the
existing trained models and negative results. Check A2, F2 and diagnostic release
hashes before execution; run scientific analysis/fitting only on Slurm compute.

Report H=1, H=2 and continuation increments H2-H1; within-parent head/utility
variance and covariance; H1/H2 rank agreement; strict winner-set reversals;
immediate-oracle future regret with uniform tie averaging; winner margins;
exact ties; descriptive margins within 1/768 and 1/192 utility units; terminal
availability; and negative utility at 0, -1/768, -1/192 and -0.02. The diagnostic
resolution thresholds do not replace any gate or define statistical harm.
Break down training/development by model/task stream and generation. Audit
feature availability, global variation and within-parent variation.

Use fixed original independent GRU architecture and F2 independent fitting
rules, for immediate, future and continuation targets. Train using nested
whole-trajectory budgets 5/10/15/20 and seeds 42/43/44: 36 fits. Construct one
round-robin stream order using hash-sorted trajectory IDs with seed42, without
looking at outcomes. Use train-only normalization and H1 consequence scales.
Keep AdamW LR0.001, decay0.0001, whole-state batch16, clipping1, max200 epochs,
patience40, F2 comparative loss and own-target checkpoint score unchanged.
Continuation models predict increments only and are explicitly diagnostic;
their predictions cannot serve as full H2 values for an online selector.

Include absolute and candidate-relative ridge regression with fixed penalty1
for all three targets and four sample budgets: 24 deterministic fits. Use only
the existing pre-commit input allowlist, train-only feature standardization,
and no validation-selected regularization. Include uniform tied/random and
fixed lowest-pre-update-NLL controls. Immediate-oracle selection uses observed
H1 labels and is explicitly privileged, not an available pre-commit baseline.

Report per-state ranking, per-head errors, top1 with tie-adjusted chance, and
whole-trajectory bootstrap intervals, 5000 draws/seed42. Average fitting seeds
within each state before paired comparisons. Predeclare three paired Spearman
contrasts on full-data GRU fits: immediate-own minus future-own; future-own
minus immediate predictions scored on future; immediate-own minus
continuation-own. Apply Holm adjustment to their three paired permutation
tests. No favorable seed, feature, subgroup or training size replaces a primary
result. Development observations are exploratory, not final independent tests.

This study can identify a horizon gap, weak feature variation or a learning
curve trend. It cannot establish irreducible noise, a universal predictability
ceiling, terminal-state contrasts when no terminal examples exist, an
architectural improvement, or recursive-policy benefit. Stop at the review
report and record runtime, failures and source/label hashes. Any confirmatory
test or GPU continuation-repeat study requires its own frozen scope.
