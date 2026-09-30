# Third Eye Research

**Third Eye: Forecasting the Long-Term Value of Self-Updates for Recursive
Self-Improvement in LLMs.** This repository implements the first sprint's
systems/training component: verified self-corrections, matched LoRA/QLoRA
candidates, checkpoint rollback, H=1/H=2 label generation, and recursive job
execution. The research proposal is in
[docs/ACL27_Third_Eye_Research.docx](docs/ACL27_Third_Eye_Research.docx).

## Current status

The modular training infrastructure and local tests are implemented. Real
HF/PEFT LoRA training is verified on a tiny random Llama model on CPU. A
Qwen3-4B GPU pilot, NF4 runtime verification, measured hyperparameter freeze,
and empirical Gate 1 findings are **pending**. No benchmark result, ranking
reversal, or performance improvement is claimed by this commit.

The supplied Qwen/Llama/Gemma configs are starting **pilot** configurations;
their LR, rank, candidate size, and step count are not empirically frozen.
The forecasters and official benchmark preparation/evaluation are separate
components to integrate through [docs/integration.md](docs/integration.md).

## Layout

| Directory | Responsibility |
| --- | --- |
| `src/third_eye/training/` | HF/PEFT loading, masked SFT, optimizer loop, adapter checkpoints |
| `src/third_eye/updates/` | Failure collection, verified corrections, equal-size stratified candidates |
| `src/third_eye/data/` | JSONL contracts, immutable manifests, overlap checks |
| `src/third_eye/evaluation/` | Development/proxy scores and verifier interface |
| `src/third_eye/experiments/` | Reversible probes, branch labels, recursive orchestration |
| `experiments/configs/` | Pilot configs for Qwen3-4B, Llama-3.2-3B, Gemma-3-4B |
| `experiments/jobs/` | SLURM pilot and GPU-verification jobs |
| `tests/` | Data integrity, candidate isolation, H=2 semantics, real LoRA integration |
| `docs/` | Research proposal, integration contract, runbook, validation record |
| `data/` | Synthetic interface fixtures; real datasets stay untracked |
| `models/` | Model-artifact instructions; weights stay untracked |

## Install and verify locally

Use Python 3.10–3.12. There is no `pyproject.toml` or package-build step.
Entrypoint scripts add `src/` to their import path; for your own scripts use
`PYTHONPATH=src` or add that directory to your interpreter path.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m ruff check src experiments tests
python experiments/verify_backend.py --device cpu
```

For a CPU-only machine, install CPU PyTorch before the requirements:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For NF4 QLoRA on a CUDA host:

```bash
pip install -r requirements-gpu.txt
python experiments/verify_backend.py --device cuda
python experiments/verify_backend.py --device cuda --quantization nf4
```

The verification script creates a tiny model locally; it downloads no model
weights. It checks actual parameter updates, frozen base weights, exact
rollback, checkpoint round trips, cumulative updates, seed repeatability,
and generation. Its outputs are software checks, not research evidence.

## Prepare splits and run a pilot

Obtain four independently audited JSONL splits from the evaluation component:
`train`, `target_dev`, `ood_dev`, and `retention_dev`. Read
[data/README.md](data/README.md) for the schema. Training and selection never
consume final test sets. The tiny files below only demonstrate manifest IO:

```bash
python experiments/make_manifest.py \
  --train data/examples/train.jsonl \
  --target-dev data/examples/target_dev.jsonl \
  --ood-dev data/examples/ood_dev.jsonl \
  --retention-dev data/examples/retention_dev.jsonl \
  --output data/manifests/synthetic.json
```

With real audited development splits and an allocated GPU:

```bash
python experiments/run.py \
  --config experiments/configs/qwen3_4b_pilot.json \
  --manifest data/manifests/math_v1.json \
  --output runs/qwen_math_pilot_seed42 --pilot
```

Math verification defaults to a strict numeric GSM8K-style verifier. For
MBPP/code or richer math/retention tasks, supply a trusted verifier factory:
`--verifier your_evaluation.verifiers:make_verifier`. Code verification must
execute in the evaluation component's sandbox; this harness does not execute
generated programs on the host.

After measuring and choosing the pilot settings, freeze the config that
actually produced a completed pilot:

```bash
python experiments/freeze_protocol.py \
  --config experiments/configs/qwen3_4b_pilot.json \
  --pilot-run runs/qwen_math_pilot_seed42 \
  --output experiments/configs/qwen3_4b_frozen.json
```

Run the frozen protocol without `--pilot` to generate larger meta-label slices.
Each invocation uses a new output directory. Set `protocol.depth` before the
pilot/freeze to choose the number of accepted updates per invocation; the
runner enforces T≤5. Start with T=1 for the pilot. Resume with `--resume` using
an `accepted/generation_N/` checkpoint and the same config, manifest, and
policy. See [docs/runbook.md](docs/runbook.md) for SLURM and resume commands.

## Experimental guarantees

- K=3 candidates have equal sizes, identical difficulty quotas and optimizer
  budgets. All start from the exact same cumulative parent adapter.
- Verified candidate targets are self-generated by the current model. Gold
  answers are withheld from initial and correction generation prompts.
- Short probes are rolled back before full updates. AdamW is reset for every
  update; accepted adapter changes persist across generations.
- H=1 and H=2 consequence vectors contain separate target, OOD, and retention
  deltas **from the parent**, in accuracy fractions. Fixed utility weights
  derive scalar labels; the three outcomes are retained independently.
- H=2 uses one standardized continuation procedure: same training pool, seed,
  sampling rule, batch size, and optimizer budget across branches. Verified
  continuation completions depend on each evolved t+1 model.
- Only the selected **t+1** adapter is committed. The t+2 adapter is label
  evidence. External selectors receive pre-commit features without labels.
- If a branch cannot produce enough verified corrections, the state fails and
  rolls back. No fabricated continuation, silent batch-size change, or partial
  K-state enters the meta-label dataset.
- 8B runs require a documented Gate 2 decision. PD, RLHF, and full-model
  fine-tuning are outside this sprint.

## Outputs and handover

`runs/<run>/meta_labels.jsonl` is the forecaster input/label interface.
`run.json` records config, manifest, seed, source commit, and base revision.
`ledger.jsonl` records completed states, acceptance decisions, and runtime.
`states/` contains correction pools, candidate batches, probes, train logs,
H=1/H=2 adapter checkpoints, and per-candidate records. `accepted/` contains
cumulative adapters and resume metadata.

Keep `labels`, post-update evaluations, and full-update training logs out of
forecaster inputs. Split meta-training/validation/test **by recursive state
and trajectory**, never by candidate row. All three candidates from one state
must remain together. The detailed integration contract and remaining sprint
work are documented in [docs/integration.md](docs/integration.md).
