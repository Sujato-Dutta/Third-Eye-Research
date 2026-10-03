# Third Eye implementation contract

## Inputs and labels

Candidate records contain `state_id`, `trajectory_id`, `candidate_id`, generation,
model/seed identifiers, immutable protocol/data hashes, and cumulative adapter
hashes. `precommit` contains current development scores, three past accepted
state summaries, verified-pool statistics and reversible candidate diagnostics.
`labels.h1` and `labels.h2` hold separate target/OOD/retention changes from the
parent in accuracy fractions. Fixed utility weights derive scalar values.

`forecasting.encoding` is a versioned allowlist. IDs, model identity, branch
checkpoints, actual continuation data, final tests, full-update logs, and labels
never enter inference. Missing optional diagnostics have explicit availability
bits. Normalization is fitted only to meta-training data.

Meta-splits keep entire recursive trajectories together, merge trajectories
sharing a state, preserve all K=3 candidates, reserve 20% of independent core trajectories each
for validation and test, and reserve specified held-out
models for test. Resumed runs preserve trajectory IDs. Legacy records without
IDs derive their trajectory from the run/resume provenance or source directory.

## Features and architectures

The HF backend measures tokenizer lengths, mean completion likelihood,
mean-input-embedding diversity, adapter gradient norms, cosine/sign agreement
with eight retention examples, next-token KL on eight fixed anchor prompts,
and adapter displacement. Gradient features use the first eight candidate
examples and summarize layer norms without model-specific layer indices.
Probes use the frozen short-step budget and always restore the parent adapter.
Word and token counts remain distinct measurements.

OneStep is a two-hidden-layer MLP. Direct uses a two-layer GRU over masked
history and a multi-head feed-forward decoder. `matched_h1` uses the same Direct
architecture and features, changing only the supervised horizon. Dynamics
encodes current capability/history and candidate features, applies a shared
residual transition twice, and predicts the second update embedding from the
predicted first latent state. The actual continuation is unavailable online.
H=1 decoding and detached capability-state encodings supply auxiliary training
supervision; post-update scores are used only as targets. This is a latent
model-evolution forecaster, not a general environment world model.

Training combines consequence MSE and within-state pairwise ranking loss.
Dynamics adds intermediate decoding and latent consistency losses. Dynamics reports H=1/H=2 latent rollout
error against encoded future capabilities with fixed current-history context.
Validation
selects checkpoints and stopping; test data is evaluated after selection. All
forecasters stay below one million trainable parameters. Scalar H=2 and
history/gradient/probe/diversity/retention ablations reuse the same split.

## Execution and evaluation

`run.py --mode labels` computes all K=3 H=1/H=2 branches and commits only the
selected t+1 adapter. The continuation generation/sampling/training procedure
and seed are fixed across branches; completions depend on each evolved model.
Incomplete states are never published. Insufficient verified corrections fail
rather than silently changing the batch or fabricating a label.

Candidate sampling first draws a seeded difficulty quota. If that quota cannot
support K distinct batches, a count-based search finds the quota with maximum
combination capacity at the same batch size. Feasible original quotas retain
their seeded behavior. All candidates share the selected quota; the search
uses no consequences or held-out scores. Truly insufficient matched pools fail.

`--mode online` computes matched candidate features, ranks before commitment,
and trains one selected full update. Greedy evaluates three real H=1 branches;
no-update leaves parameters fixed. Online runs log actual end-to-end wall time
and probe time separately. Feature probes are paid by all update policies for
matched comparisons. Failure restores the current parent. Final tests are
loaded only by `evaluate_final.py`, which writes item correctness separately.

GSM8K numeric verification is strict. MATH uses the pinned Math-Verify parser
and equivalence checker. MMLU prompts request one answer letter. MBPP prompts
include required function signatures derived from the reference interface,
without unit-test answers; this prompt protocol must be declared in reports.
HumanEval accepts complete functions or reconstructed body continuations and
reports greedy pass@1. The code development proxy is MBPP validation, not
HumanEval and not a separate cross-benchmark OOD claim.

## Isolated code verification

Set `THIRD_EYE_SANDBOX_CONFIG` to an untracked JSON file:

```json
{
  "engine": "docker",
  "image": "python@sha256:<actual-approved-image-digest>",
  "timeout_seconds": 8,
  "memory_mb": 256
}
```

Pull/approve the image through the cluster's normal provisioning process.
The verifier requires an operational Docker daemon in the allocated job.
Containers have no network, read-only inputs/root filesystem, no Linux
capabilities, a non-root UID, and PID/memory/CPU/file/output limits. Timeout and
excess output terminate the container. Tests are held outside the generated
prompt. A sandbox startup failure fails the run rather than scoring every
example incorrect. The verdict nonce rejects ordinary output and early exit;
the benchmark harness is not a defense against deliberate test introspection.

Bubblewrap is also supported with `engine: bubblewrap`, an explicit executable
path and SHA-256, and a Python virtual-environment runtime. It mounts only the
read-only system runtime, virtual environment, and invocation files; clears
inherited environment variables; and separates user, PID, network, IPC and
mount namespaces. CPU, address-space, process, file and output limits apply.
Timeout kills the process group, and namespace startup must succeed. The
configured memory bound is virtual address space, not a cgroup RSS limit.
The DGX preflight additionally checks reference programs before code pilots.

If these backends are unavailable, use a trusted cluster-approved verifier with the
`--verifier module:factory` interface. The callable receives `(example,
completion)` and returns a boolean; its exceptions propagate. Never substitute
host `exec`, a plain subprocess, or a network/home-mounted Enroot session as the
isolation boundary for generated programs.

## Evidence and gates

Ranking reports use tie-aware top-1, Spearman/Kendall and NDCG with within-state
shifted nonnegative gains. Undefined correlations are null. Ranking utilities are rounded to 12 decimal
places to prevent machine-rounding differences from creating false reversals. Confidence
intervals resample states, with trajectory-cluster intervals additionally
reported for the phenomenon. Policy comparisons pair model, split, seed,
revision and budgets; final item tests pair IDs within each benchmark role.
Holm correction is applied across reported comparisons. Per-head errors,
calibration bins and zero/nonzero sign accuracy remain separately visible.

Gate 1 requires at least 20 complete states and >=20% strict ranking reversals
or >=15% updates harmful on a consequence. Gate 2 uses validation, requires
at least 30 states with defined within-state correlations, mean Spearman >=.30,
and informative-state top-1 >=.45 and above its tie-adjusted chance rate.
The minimum sample counts are conservative implementation safeguards beyond
the numerical signal thresholds. Eight-billion-parameter runs require the
measured Gate 2 artifact. PD remains unimplemented until its additional
retention/trade-off activation criteria are measured and passed.

All runs log source file hashes, dependency versions, model revision, split and
adapter hashes, generation/seed and actual runtime. Result reports exclude
pilot/incomplete trajectories and reject duplicated policy/seed units.

`THIRD_EYE_MODEL_SOURCES` can point to a mapping of backbone IDs to immutable
public-source proofs. `prefetch_mirror.py` requires every downloaded weight
shard to match the original repository's SHA-256 and the tokenizer vocabulary
to match its original content hash. Exact matching metadata can be recovered
from another pinned public source. Configuration and chat/generation metadata
differences remain explicit in the proof; the entire mirrored package is not
claimed to be byte-identical to the original. Backend checkpoints enforce the
same source proof on reload, and paired reports include its hash.

`protocol.generation_batch_size` fixes left-padded greedy inference batches.
Sampled correction attempts retain individual seeds. Batch size is part of
the protocol fingerprint when greater than one, and the single-prompt default
preserves earlier fingerprints. Pilots measure the chosen inference batch size
before freezing; all policy comparisons for that backbone use it.

`protocol.sampled_batch_size` fixes batched correction retries. Every training
example/attempt retains its own seed and random stream, with the same
temperature, model top-k and top-p=0.95 transformations. Successful examples
leave later retry rounds; the saved correction pool retains training-example
order. Fixed-batch repeatability is checked; floating-point batching effects
can change tokens compared with serial sampling. A non-default sampled batch
size enters the config hash, and its default preserves legacy fingerprints.
CPU verifier work is separate from model generation. Set
`THIRD_EYE_VERIFIER_WORKERS` within `SLURM_CPUS_PER_TASK` for independent code
sandboxes; `THIRD_EYE_SYMBOLIC_WORKERS` defaults to one. Ordered verdicts and
infrastructure exceptions are preserved. Jobs record CPU/GPU resources,
verifier settings and sandbox-config hashes, plus pool stage timings.

`protocol.multiple_choice_scoring: conditional_likelihood` ranks A/B/C/D by
the sum of conditional answer-token log probabilities under the same chat
prefix. It excludes end tokens and never conditions on the reference answer.
Retention questions use this fixed accuracy protocol to avoid explanation
generation and answer-formatting confounds. Math/code still use greedy
generation. The setting is part of the protocol fingerprint, is fixed before
research label collection, and must match across compared policies. The
default `generation` setting preserves earlier protocol fingerprints.
