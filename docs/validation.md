# First sprint local validation

Validated on September 30, 2026 using Python 3.12, CPU PyTorch 2.8.0,
Transformers 4.56.2, PEFT 0.17.1, and Accelerate 1.10.1.

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 29 tests passed |
| `python -m ruff check src experiments tests` | Passed |
| `python experiments/verify_backend.py --device cpu` | Passed |
| JSONL fixture manifest creation and readback | Passed |
| Experiment entrypoint help/config parsing | Passed |
| `bash -n` on both SLURM scripts | Passed |

The real HF/PEFT checks use a tiny randomly initialized Llama on CPU. They
verify actual LoRA parameter changes, unchanged frozen base weights, exact
rollback and checkpoint reload, cumulative updates, fixed-seed reproduction,
assistant/padding loss masking, and text generation. A separate integration
test exercises actual probes, LoRA updates, and checkpoints through all K=3
H=2 branches with deterministic fixture generation.

Orchestration tests verify candidate batch/stratum matching, equal budgets,
parent isolation, H=2 deltas measured from the parent, rejection of partial
states, rollback on exceptions, training-only corrections, immutable split
hashes, cross-split leakage rejection, and exclusion of future labels from
external-selector inputs.

No GPU run was performed in this environment. NF4 bitsandbytes, Qwen3-4B,
Llama-3.2-3B, and Gemma-3-4B runtime behavior still require verification on the
actual cluster. The prepared SLURM smoke job enables that next check. The real
pilot and empirical hyperparameter freeze remain outstanding; test fixtures
are software evidence only and provide no research benchmark results.
