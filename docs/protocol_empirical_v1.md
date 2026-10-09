# Empirical supplement E1: reliability and balanced replication

Declared October 8, after authorization to begin the deadline-bounded empirical
package. Existing A2 labels, optimizer budgets, harvesting budgets, gates and
negative results remain unchanged. Scientific computation stays on Slurm nodes.

Select twelve parent states from the immutable snapshot's training/development
partitions, three per model/task stream at generations 0, 2 and 4, with distinct
trajectory IDs. Choose by the fixed hash ordering in the released sampler;
selection never uses outcomes. Preserve the entire chosen sample and any failure.

Completed branch weights were pruned by the original storage policy. Rebuild
ancestors and candidate/continuation adapters from preserved verified batches
and original training seeds, without harvesting those pools again. Require exact
original parent, M1 and M2 adapter hashes before new measurements. A failure is
an implementation/reproducibility review stop, never an excuse to alter seeds or
replace the state. Keep rebuilt and new checkpoints in an isolated namespace.

Record aligned item IDs, prompt hashes, prediction hashes and verifier outcomes
for original parent/M1/M2, one alternative optimizer seed with the fixed candidate
batch, one alternative optimizer seed with the original continuation batch, and
two fresh standardized continuation seeds from fixed M1. Alternative optimizer
seed is original+50,000; fresh continuation seeds are original+50,000,000 and
original+100,000,000. Each alternative is shared across the three candidates
within the same parent, retaining matched randomness. Fresh harvesting uses
eight attempts, the original training portion, unique samples, size2 and fifty
optimizer steps; scarcity uses the unchanged terminal M2=M1 rule.

Keep numerical repeatability, finite-item uncertainty, fixed-batch optimization
variation and stochastic future-data variation separate. Bootstrap aligned item
outcomes across candidates/horizons within each role, then cluster parent-level
effects by trajectory. Report winner stability, original-to-repeat disagreement,
original H1 versus seed-averaged H2 ranking, margins, selection value and regret.
Do not substitute averaged labels into the original single-seed Gate 1/2 data.

Cached analyses compare immediate-best, random, future-best and holding model
parameters fixed over both steps. Oracles average exact ties. Report absolute
future utility and target/OOD/retention heads, not only forced-update regret.
Holding may outperform all candidates. Predeclare four exploratory associations:
probe-magnitude dispersion, mean retention-gradient cosine, mean data diversity,
and candidate-loss dispersion versus observed immediate-best future regret.
Use trajectory uncertainty; do not claim causality or choose forecaster inputs
from these results. Synthetic positive controls use a known within-parent loss
signal with fixed noise levels 0/.5/1 and seeds42/43/44; label them synthetic.

Prepare eight balanced new T5 label trajectories, two seeds6142/6143 per stream,
retaining all branch checkpoints for reliability analysis. They constitute a
fresh held-out replication cohort. Freeze methods/comparator before outcomes;
do not fit, select seeds, tune features or change scope based on these labels.
Release this cohort only after reviewing preliminary noise evidence. No main
online/scaling experiment follows, and no 8B model is introduced.

The known-signal control does not establish real-data power. A TuneAhead-style
tabular comparator uses only available pre-commit descriptors/probe inputs and
fixed published small-tree fitting settings; it is an adaptation, with ten-step
probes instead of the paper's hundred. It is trained/development-selected only
on the old training/development partitions and confirmed on the fresh cohort.
The old exposed test cannot become its final untouched test.

All source/scope/input files are hashed. Full CPU tests and exact real-model
GPU replay verification precede the twelve measurement jobs. All user GPU
jobs, including three recoveries, count toward a rolling ceiling of twenty.
Stage reviews are explicit; no new success threshold or favorable-state
replacement is invented. Aim to freeze results by October11 23:59 IST.
