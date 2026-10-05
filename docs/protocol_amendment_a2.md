# Protocol Amendment A2: Eight-attempt correction harvesting

Frozen before new A2 experiments on October 5, 2026, following explicit
authorization to use eight correction attempts for all subsequent runs.

The only scientific budget change from A1 is a maximum of eight correction
attempts per initially failed training problem, stopping at the first verified
success. This applies to parent and every standardized H=2 continuation harvest,
all label generation and online policy trajectories. The full predeclared
1,024 GSM8K / 306 audited MBPP training prompts remain scanned. There is no
selection based on final-test performance and no replacement with test data.

K=3, candidate size two, unique examples within batches, distinct compositions
with overlap across candidates, rank 16, alpha 32, learning rate 0.0001,
fifty optimizer steps, probe ten steps, and all other A1 training/evaluation
budgets remain unchanged. The A1 terminal rule remains: fewer than two verified
continuation corrections means M2=M1, availability zero, zero continuation
optimizer steps and equal H=1/H=2 consequences. Parent composition scarcity
ends the trajectory without publishing incomplete labels or duplicating data.

Correction draws retain a seed stride of sixteen per training-prompt index:
`harvest_seed + 100000 + index * 16 + attempt`, with attempts zero through seven.
This matches the tested eight-attempt prefix instead of changing random draws
for every problem when the cap is reduced. The algorithm, prompts, verifier,
maximum 512 new tokens, temperature 0.7 and inference batch sizes are unchanged.

CPU and CUDA implementation verification, plus two independent allocated-node
reproducibility controls, precede label generation. A2 enables strict PyTorch
deterministic algorithms, cuBLAS workspace `:4096:8`, disables TF32 and cuDNN
benchmarking, and enables deterministic cuDNN. Unsupported nondeterministic
operations fail validation. These runtime settings are part of the recorded A2
execution environment; historical A1 outputs need not equal new A2 outputs.
Cross-node controls must agree on completions, verifier outcomes, adapter hashes
and repeated training. No root cause of the historical discrepancy is claimed
without evidence. Determinism is scoped to the tested pinned software/hardware.

A1 and pre-A1 artifacts and source are archived unchanged. A2 has its own source,
configurations, run directories and protocol marker; datasets reject pooled
amendments. A1 Gate 1 remains screening evidence. Report the A2 phenomenon and
scarcity/terminal counts separately from A1; shorter-budget learnability and
recursive improvement must be established from new A2 labels and trajectories.

The core seed grid, three repetitions per seed, two models, both task families,
T<=5, trajectory-disjoint splits, H=1/H=2 matched comparison, Direct/Dynamics,
retention metrics and Gate 2 thresholds remain unchanged. The core plan contains
36 label trajectories (up to 180 complete states), 18 CPU forecaster fits and
96 core online policy trajectories. No 8B or transfer execution before Gate 2.

Up to twenty Vista GPU nodes may run concurrently. Scientific execution and
validation use Slurm compute allocations only. A 1,000 GPU-hour planning target
and 3,000-hour ceiling remain distinct; reservations and actual spending are
recorded separately. This amendment does not promise full-study savings from
the two root-state timing comparisons or equivalent recursive outcomes.

Determinism implementation follows the
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html).
