# Training component handover

This first-sprint implementation owns systems/training. It does not implement
the H=1/H=2 forecasters, Direct/Dynamics architectures, statistical toolkit,
official dataset splits, final benchmark scoring, or PD selector.

## Public interfaces

| Interface | Purpose |
| --- | --- |
| `third_eye.config.Config` | Immutable model/training/protocol configuration |
| `third_eye.training.base.TrainingBackend` | Model-independent orchestrator contract |
| `third_eye.training.hf_backend.HFBackend` | Single-device HF/PEFT LoRA and NF4 QLoRA |
| `third_eye.updates.corrections.collect_corrections` | Current-model failures → verified training completions |
| `third_eye.updates.candidates.sample_batches` | K=3 matched candidate composition |
| `third_eye.experiments.labeling.LabelGenerator` | Parent → three t+1/t+2 labeled branches |
| `third_eye.experiments.labeling.run_trajectory` | Commit selected t+1 adapter and repeat |
| `third_eye.evaluation.verifiers.VerifierRegistry.from_spec` | Inject official verifier implementation |

## Candidate record schema v1

Each record has identifiers (`state_id`, `candidate_id`, `generation`, `model`,
`seed`), protocol/split/adapter/batch hashes, and these groups:

| Field | Contents | Allowed in forecaster input? |
| --- | --- | --- |
| `precommit.state` | Current target/OOD/retention development accuracy | Yes |
| `precommit.history` | Last three accepted state summaries and H=1 changes | Yes |
| `precommit.features` | Candidate word-length/duplicate/difficulty/NLL signals and reversible probe changes | Yes |
| `precommit.pool_statistics` | Current training-pool failures and correction verification rates | Yes |
| `labels.h1` | Immediate target/OOD/retention changes from the parent | Targets only |
| `labels.h2` | Future target/OOD/retention changes from the parent | Targets only |
| `labels.utility_h1`, `labels.utility_h2` | Fixed weighted aggregates of consequence heads | Targets only |
| `evaluation`, `runtime`, `continuation`, `checkpoints` | Branch audits and post-update artifacts | No |

Metric values are fractions: `0.02` means two percentage points. Positive
retention delta means improvement; forgetting is its negative. Baseline
accuracies and batch-derived inputs are logged before full candidate training.
The current length signals count words, not tokenizer tokens. Do not silently
interpret them as token features. Token lengths, embedding diversity,
confidence, retention-gradient cosine/sign agreement, KL drift, and richer
validation-loss features belong to the modeling feature extractor.

The systems logger already emits total and layerwise adapter gradient norms
at every optimizer step in `probe.jsonl` and `training_t1/t2.jsonl`. Probe logs
are pre-commit; full-update logs are post-update. Keep this distinction when
building additional features. The default probe records a per-step loss
change and the retention-proxy NLL change; it is a starting logging adapter,
not the final frozen modeling feature schema.

## Plug in modeling features

Inject a function with the default extractor signature:

```python
def features(backend, batch, retention_dev, protocol, seed, log_path) -> dict:
    ...

labeler = LabelGenerator(
    backend, config, splits, verifier, output, manifest_hash,
    feature_extractor=features,
)
```

The extractor must restore the parent adapter even on exceptions. The labeler
checks the parent hash afterward. `backend.snapshot()` returns a CPU copy of
the cumulative adapter; `restore()` clears gradients. Use a `try/finally` guard
around any probing. Do not merge adapters into the frozen base model.

## Plug in a learned selector

```python
def choose_candidate(views):
    # Each view contains state_id, candidate_id, and precommit only.
    predictions = forecaster.predict([v["precommit"] for v in views])
    return int(predictions.argmax())

run_trajectory(labeler, policy="external", selector=choose_candidate)
```

The default meta-data exploration policy is seeded random. `greedy_h1` is an
explicit experimental baseline that pays for candidate updates and uses the
measured immediate development score. Never use measured H=2 labels to choose
an online policy. Label generation is expensive research data collection;
external-policy execution here still computes H=2 labels for auditing. A later
deployment-only fast path should probe/rank candidates and train only the
selected one, and must report its own runtime separately.

## Evaluation integration

The external verifier factory returns `callable(example, completion) -> bool`.
It is authoritative for all tasks in that run. Route math/code/retention tasks
inside that callable, using `example.task` and metadata. Exceptions propagate
to fail the state; the systems runner never treats a verifier crash as a valid
correction. Generated code must run with the evaluation team's actual sandbox
and resource limits. The bundled numeric verifier is for numeric GSM8K-style
answers only; fractions, symbolic math, units, and rich MATH scoring require
the official evaluator.

The evaluation component supplies source-disjoint training/development data,
fixed retention proxy/anchor, and independent final test data. The original
proposal's final MATH-500/HumanEval scores remain evaluation-only; they must
not be fed into the online feature pipeline. Audit overlap beyond exact text.

## Pilot and freeze checklist

The real GPU pilot still needs to establish throughput/memory, enough verified
corrections, sensible LR/rank/steps/batch size, and reproducible K=3 H=2 labels.
Only then freeze those settings. Gate 1 ranking mismatch and harmful-update
rates must be measured from real states, not the test fixtures. Llama-small
jobs are prepared; Qwen/Llama/Gemma need their usual model access (including
gated-model approval where applicable). Pin HF revision to a commit before
long research runs and archive dependency/runtime logs. No 8B scaling before
Gate 2. No claim that the first sprint's empirical milestones are complete.
