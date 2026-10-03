# Software and DGX validation

Validated on October 1, 2026 using Python 3.12, CPU PyTorch 2.8.0,
Transformers 4.56.2, PEFT 0.17.1, and Accelerate 1.10.1.

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 74 tests passed locally |
| `python -m ruff check src experiments tests` | Passed |
| `python -m ruff format --check src experiments tests` | Passed |
| Python compilation | Passed |
| Experiment entrypoint help/config parsing | Passed |
| `python experiments/verify_backend.py --device cpu` | Passed |
| `python -m pip check` | Passed |
| `git diff --check` | Passed |

The real HF/PEFT checks use a tiny randomly initialized Llama on CPU. They
verify actual LoRA parameter changes, unchanged frozen base weights, exact
rollback and checkpoint reload, cumulative updates, fixed-seed reproduction,
assistant/padding loss masking, and text generation. Integration checks
exercise probes, retention gradients, anchor KL, LoRA updates, and checkpoints
through matched K=3 H=2 branches using deterministic fixture generation.

Pipeline tests cover all forecaster architectures, trajectory-isolated splits,
train-only normalization, feature allowlists, model holdouts, checkpoint
selection, held-out predictions, and learned online selection. They also check
equal candidate budgets, parent isolation, deltas measured from the parent,
rollback on exceptions, training-only corrections, portable manifest hashes,
split overlap rejection, and resumed trajectories retaining their baseline.

Evaluation checks cover numeric, symbolic, and multiple-choice verification;
benchmark preparation uses mocked source rows and pinned revision responses.
Code-sandbox command construction and failure handling use mocked Docker
processes. Final-evaluation tests verify sealed split isolation and linkage to
the selected development checkpoint. Actual symbolic equivalence is checked
with Math-Verify, including its bounded Windows subprocess wrapper.

Statistical and orchestration tests cover tie-aware rankings, calibration,
paired inference, gate criteria, result tables and plots, source provenance,
storage pruning, scheduler reservations, and compute-node guards. Synthetic
report fixtures exercise development trajectories and forecast consequences.

No research-model training was performed locally. Scheduled DGX jobs completed
environment provisioning and both official benchmark-suite preparations.
Compute-node CUDA LoRA and NF4 training, rollback, checkpoints and reproducibility
checks passed, alongside the original 61 tests. The expanded 74-test suite also
passed on DGX. CUDA/NF4 batch generation passed repeatability and adapter-isolation
checks. Qwen3-4B generated eight 64-token completions in 6.62 seconds using
4.01 GB peak allocated CUDA memory; this is a throughput smoke check, not a
benchmark-accuracy result.

Conditional multiple-choice scoring matches independently calculated full
logits in the local tests. Scheduled Qwen3-4B NF4 scoring passed repeated and
batched/individual prediction agreement, adapter isolation and padding
restoration. Eight real retention prompts took 1.10 seconds with 4.32 GB peak
allocated CUDA memory. Neither throughput check estimates benchmark accuracy.

Full-A100 profiles passed on `dgxb`. Per-row seeded sampling matched serial
sampling and reordered rows on a real CUDA/NF4 tiny model. Real Qwen NF4 sampled
completions matched serial results in the eight-prompt check. Fixed bf16 batches
passed repeatability and adapter isolation, plus an actual LoRA optimizer step
and exact rollback. Quantization and batching may change outputs; their settings
remain part of each fixed comparison protocol. The bf16 training smoke used
9.09 GB peak CUDA memory. CPU parallel code verification preserved all 252
canonical reference verdicts and took 3.26 seconds with 16 workers versus 19.17
seconds serially. Symbolic verification remained serial after its parallel
profile was slower. See the execution record for measured generation times.

Candidate regression tests cover an infeasible initial random difficulty quota
with an otherwise sufficient pool, and truly insufficient matched strata. The
exact failed DGX pool (two short and three long corrections) passed the fixed
sampler: three distinct size-two batches share identical difficulty counts.

The Bubblewrap namespace probe passed on a compute node. Its verifier passed
correct/incorrect-program, early-exit, timeout and host-filesystem checks and
all 252 prepared MBPP training reference programs. This is infrastructure
verification, not model code-generation accuracy. The local Docker engine was
unavailable; Docker execution remains unverified. SLURM scripts passed Bash
syntax checks on DGX. All research-model pilots and measured gates remain
pending their completed artifacts.

These checks establish software behavior only. They provide no benchmark gains,
forecasting gates, transfer results, or research conclusions. The optional
primal-dual extension remains gated on measured forecasting and retention
signal, as specified in the research protocol.
