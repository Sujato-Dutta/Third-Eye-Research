# Signal confirmation v1

Declared October 8 after development diagnostic v1 and before this confirmation's
held-out evaluation. User authorized deadline-bounded experiments to decide the
paper direction. This freezes an empirical-analysis confirmation, not an
architectural recovery or a revision of A2/F2/Gate 2.

Development found weak continuation rankings and non-monotonic H2 learning
curves. The fixed empirical hypotheses are that immediate and future winner
sets differ and that continuation-value forecasting is weaker under the tested
feature/model protocol. Hypotheses are conditional; no irreducible-noise or
universal-impossibility claim is tested. A positive finding needs uncertainty
and appropriate limitations; weak confirmation must be reported.

Use the existing snapshot's forty test states from eight trajectories, disjoint
from the twenty training and eight development trajectories. The original
label campaign previously reported aggregate Gate 1 across its published
population, and original forecaster jobs generated test metrics; those facts
must be disclosed. This confirmation is not a newly collected population.
No test forecasting result is used to select or modify the methods below.

Evaluate all nine already-fitted full-training independent GRUs: immediate,
future and continuation targets, seeds42/43/44. Keep their exact weights,
training normalization and development-selected checkpoints. Validate prediction
reload equality on development before evaluating held-out labels. Do not
evaluate/select reduced-training-size winners. Include all six fixed
full-training ridge controls, penalty1, absolute/relative inputs and three
targets, plus uniform tied/random and lowest-pre-update-NLL controls. Ridge
coefficients are deterministically reconstructed from training only with the
already frozen specification; there is no tuning or new learned architecture.

Report the same signal decomposition, utility/head rankings, stream/generation
breakdowns and trajectory bootstrap intervals as diagnostic v1. Average the
three fitting seeds within each state before three paired Spearman comparisons:
immediate-own minus future-own, future-own minus immediate scored on future,
immediate-own minus continuation-own. Apply Holm correction across those three
tests. Their direction and definitions match the frozen development diagnostic.
No favorable seed, subgroup or metric replaces this fixed comparison.

Perform model inference/statistics only in a Slurm CPU compute allocation.
Check original A2/F2, diagnostic and confirmation hashes. Save scope, weight
hashes and test identities before numerical held-out evaluation. Preserve all
results and stop for review. Gate 2 remains failed on development; a favorable
test number does not replace its decision. No online policies, GPU harvesting,
scaling or new fitting sweep follows automatically. After this evaluation,
these test results cannot support further iterative development and then be
presented as an untouched final test for the resulting method.
