# E2: checkpoint-based balanced reliability supplement

Adopted October 8, 2026 following authorization to execute the strongest feasible
deadline package. E1's structural cold-reconstruction failure, all original
results and failed logs remain evidence. This supplement changes execution order
and its reliability sample. It does not relax historical hash checks, replace
old labels, amend A2, or reinterpret the original failed Gate 2 as a pass.

The fresh confirmation cohort is eight T=5 random-policy trajectories: seeds
6142 and 6143 in each of Llama3.2-3B math/code and Qwen3-4B math/code. All existing
methods, normalizations and the primary Direct-H2 versus matched-H1 comparison
were frozen before new outcomes. No method is chosen or fitted using this cohort.
There are 40 states and 120 candidate labels if all trajectories complete.

Reliability uses exactly eight independent parents: generation 0 for seed 6142
and generation 4 for seed 6143 in each stream. Each state must have actual saved
parent, all three M1 and all three M2 checkpoints, with matching stored hashes.
Weights are retained for every generation; no reconstruction or branch pruning.
Failed or correction-starved trajectories are reported, never replaced. There
is no newly invented terminal-frequency threshold.

For each selected parent, evaluate all three original M1/M2 checkpoints on the
same aligned target/OOD/retention items. Record differences from the native
stored totals rather than demanding or claiming exact evaluation reproduction.
For code, verify each identical completion three times to separate verifier
variability from regeneration. Prediction hashes and item identities are saved.

Controls are one fixed-candidate-batch optimizer-seed repeat from the actual
parent, one fixed-continuation-batch optimizer-seed repeat from actual M1, and
two newly harvested continuations from the same actual M1. Offsets are +50,000
for optimizer repeats and +50,000,000/+100,000,000 for fresh continuations.
All conditions retain K=3, size 2, unique examples within each batch, the frozen
LoRA/learning-rate settings, 50 optimizer steps, eight correction attempts,
stride 16, train portions 1024/306, and evaluation sizes 64/64/256. Fresh pools
may vary by seed and evolved model. All corrections must pass the fixed verifier.
Terminal continuation has zero executed steps and M2=M1. Original single-seed
labels stay unchanged; three-continuation averages are supplementary analyses.

Matched controls measure finite-item, optimizer and continuation sensitivity;
they do not establish a universal noise floor. Report reversal differences,
selection regret, greedy/random/oracle/hold-parameters values and trajectory
confidence intervals. Four streams are equally weighted in balanced forecasting
confirmation. Reliability intervals use eight independent trajectory units;
two units per stream and overlapping evaluation sets limit precision. Frozen
mechanism associations are descriptive, not causal. Synthetic controls show
detection of a constructed signal, not power for all real forecasting effects.

The native scientific implementation remains source SHA
917c86b78482c73797619d17feab423f91d3775026db086e15f7f529e4fcf988.
An operational wrapper publishes the selected completed state and submits its
control job before continuing the unchanged native trajectory. It does not
change model state, RNG, candidate selection or scientific budgets. Seed-6142
controls can therefore overlap later trajectory generations; seed-6143 controls
start after generation 4 is labeled. Actual retained weights are required.

Require the complete CPU suite and all-four-stream CUDA loading/training/
rollback/terminal checks before launch. All scientific work runs on Slurm
compute nodes. The login relay only validates small release metadata and submits
bounded scheduler commands. Count all queued/running GPU jobs, including the
three recoveries, toward the cap of 20: at most 8 trajectories + 8 controls + 3
recoveries = 19. CPU confirmation and reliability review follow automatically
with explicit incomplete-coverage reporting; no online rollout, 8B model,
architecture search or automatic scientific pass is authorized.

GPU execution stops by October 11 23:59:59 IST (18:29:59 UTC), including jobs
allocated late. Preserve incomplete outcomes. Planning estimates, not measured
E2 runtimes: approximately 174 GPU node-hours for the cohort and 53–93 for
controls, plus validation/recovery; roughly 24–40 hours for the cohort and 8–18
hours for late controls after allocation. Queue delay remains unbounded, so
completion cannot be guaranteed. Stop at review and archive all evidence.
