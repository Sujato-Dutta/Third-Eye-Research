# Local validation

Validated on October 1, 2026 using Python 3.12, CPU PyTorch 2.8.0,
Transformers 4.56.2, PEFT 0.17.1, and Accelerate 1.10.1.

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 61 tests passed |
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

No research-model training, official benchmark download, or GPU run was
performed locally. All five backbone runtimes, CUDA/NF4 behavior, actual SLURM
execution, pilot measurements, and the empirical protocol freeze require the
DGX cluster. The local Docker engine was unavailable, so real container
execution also remains unverified; code experiments require a working isolated
verifier. Local SLURM scripts were checked for LF line endings; Bash syntax
and scheduler execution require the cluster environment.

These checks establish software behavior only. They provide no benchmark gains,
forecasting gates, transfer results, or research conclusions. The optional
primal-dual extension remains gated on measured forecasting and retention
signal, as specified in the research protocol.
