# Third Eye

**Forecasting the long-term value of persistent, verified LLM self-updates.**
The project compares immediate update value with H=2 consequences, then uses
compact learned forecasters to choose LoRA/QLoRA updates over T<=5 generations.
The research design is in [the project overview](docs/ACL27_Third_Eye_Research.docx).

## Implementation

- Verified self-corrections, K=3 matched candidates, cumulative adapters,
  exact rollback, H=1/H=2 branch labels, and accepted-checkpoint recovery.
- Immutable GSM8K/MBPP development splits; MATH-500/HumanEval final-only
  evaluation; a fixed MMLU retention anchor; raw-prompt and near-duplicate audits.
- Token/NLL/confidence/diversity features, retention-gradient interference,
  reversible probes, anchor KL, adapter drift, and three-generation history.
- H=1 MLP, an architecture-matched H=1 temporal comparison, multi-head
  Third Eye-Direct, latent Third Eye-Dynamics, and feature/scalar ablations.
- Efficient online no-update/random/heuristic/greedy/learned policies; only
  label collection computes the H=2 branching tree.
- Tie-aware ranking, per-head calibration/error, grouped bootstrap intervals,
  paired permutation tests, McNemar tests, Holm correction, CSV tables and plots.
- Staged SLURM study plans, three seeds, shuffled-pool stress tests, gated
  family/scale transfer, runtime reservations, and checkpoint/cache management.

**Research results are pending DGX execution.** Local synthetic fixtures and
random tiny-model checks establish software behavior, not benchmark gains.
The primal-dual extension remains gated on measured forecasting and retention
signal. GPU execution requires a compute-node SLURM allocation.

## Local verification

Use Python 3.10-3.12. On a CPU-only development machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-evaluation.txt
python -m pytest -q
python -m ruff check src experiments tests
python -m ruff format --check src experiments tests
python experiments/verify_backend.py --device cpu
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Entrypoints add `src/`
to their import path; custom Python scripts can set `PYTHONPATH=src`.

## DGX workflow

See [the execution runbook](docs/runbook.md) for exact commands. Run setup,
benchmark preparation, verification, and training through the scheduler.

1. Submit `experiments/jobs/bootstrap.slurm` and `prepare_data.slurm`.
2. Check allocated CUDA/NF4 behavior with `verify_gpu.slurm`.
3. Run each backbone's measured pilot and freeze its hyperparameters and
   actual Hugging Face commit with `experiments/freeze_protocol.py`.
4. Generate a study plan from the frozen core configs:

```bash
python experiments/plan_study.py \
  --configs experiments/configs/qwen3_4b_frozen.json \
            experiments/configs/llama3_2_3b_frozen.json \
  --output experiments/plans/study_v1
```

5. Submit reviewed stage subsets with `experiments/submit_stage.py`. Estimates
   must come from the measured pilot. Start with enough complete states to
   assess Gate 1, then complete core label collection and forecaster fitting.
6. Inspect measured gates before transfer; run core policies, untouched final
   evaluations, and paired result reports. Keep the same splits and budgets
   across policy comparisons.

Code experiments require a working isolated verifier. The built-in Docker
backend requires a pre-pulled image pinned by digest; it has no host execution
fallback. A trusted verifier factory can integrate a cluster-approved sandbox.
See [the code verifier contract](docs/integration.md).

## Repository

| Path | Purpose |
| --- | --- |
| `src/third_eye/data/` | Schemas, portable manifests, benchmark preparation and audits |
| `src/third_eye/training/` | HF/PEFT loading, masked SFT, LoRA/NF4, diagnostics and checkpoints |
| `src/third_eye/updates/` | Failure collection, verified corrections and matched candidates |
| `src/third_eye/forecasting/` | Input allowlist, trajectory splits, architectures and fitting |
| `src/third_eye/experiments/` | Branch labels and efficient online trajectories |
| `src/third_eye/evaluation/` | Numeric/symbolic/code verification and sealed final evaluation |
| `src/third_eye/statistics/` | Ranking, calibration, inference and evidence reports |
| `experiments/` | CLIs, all five backbone pilot configs, study plans and SLURM jobs |
| `tests/` | Leakage, rollback, real tiny-model integration and pipeline contracts |

[Implementation details](docs/integration.md), [validation record](docs/validation.md),
and [data protocol](data/README.md) describe the assumptions and evidence limits.
