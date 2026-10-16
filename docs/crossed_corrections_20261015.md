# Fixed checkpoint × correction-batch diagnostic

**October16 implementation correction:** the original CUDA replay failed and
the main array never ran. The date-dependent Llama template is now under exact
replay diagnosis. The replacement design, guards and requirement to remeasure
all matrix/baseline scores are frozen in [C1](rendering_context_correction_20261016.md).
The original declaration and execution record below remain historical.

Declared October 15, 2026, after the novelty audit and before new crossed outcomes.
This is a controlled diagnostic in the current Third Eye implementation. Existing
A1/A2 semantics, historical observations and original primary results are preserved.
The eight seed-9142/9143 parents were already inspected; this is development work.

## Question and estimands

For each of the eight balanced Qwen3-4B/Llama3.2-3B × math/code parents, cross its
three saved candidate checkpoints with all three candidates' saved continuation
correction batches. Use the original three offsets 50M, 100M and 150M. Each draw
uses the same optimizer seed across all nine recipient/donor combinations.

Let Y(i,j,s) be the equal-head target/OOD/retention utility of recipient i after
training on donor j's saved batch, relative to the original parent. The primary
signed mechanism contrast is **native pairing surplus**:

    D = mean_i Y(i,i,s) - mean_(i,j) Y(i,j,s)

Average over all three declared draws and both deterministic item-half
orientations within each parent. Pure recipient or donor main effects cancel
in this balanced linear contrast. D asks whether matching each recipient to its
own generated batch helps or hurts relative to uniform donor transfer. It is
not a maximum, clipped regret, noise-corrected variance or proof of mediation.
A null D does not rule out heterogeneous interactions.

Secondary comparisons select on draws 50M/100M and one item half, then score on
150M and the other half. Compare a native-diagonal future selector, a selector
that averages every donor uniformly for each recipient, and immediate selection.
All future selectors have privileged future outcomes; no learned forecast or
deployable improvement follows. Ties receive uniform weighting at 12 decimals.
Both item-fold orientations remain repeated measurements of the same parent.

Report recipient, donor and interaction components for total future utility and
continuation gain relative to the recipient's own M1. Observed squared components
include evaluation noise and must not be called excess interaction variance.
Role-wise pairing contrasts are secondary. Parent-level intervals and paired
sign-flip tests are descriptive with eight independent development parents;
Holm-adjust all tested secondary contrasts together. Report all signs, including
negative effects. There is no automatic scientific pass or expansion threshold.

## Invariants and terminal handling

- No new harvesting. Original verified train-only batches, including order and
  completion text, must match saved batch hashes and verified pool membership.
- K=3, correction batch size two, LR1e-4, rank16, 50 optimizer steps, optimizer
  reset, accumulation and evaluation settings stay at the original configuration.
- Restore the exact saved recipient M1 before every branch. Never commit M2 to
  an accepted trajectory. Record checkpoint, batch, manifest and source identities.
- A terminal donor supplies an empty batch. Its crossed continuation is identity
  on the recipient: M2=M1_i, zero executed steps. It does not copy donor weights,
  invent examples or drop the cell. Record continuation_available=0.
- A transferred correction need not be a failure for the recipient. These are
  explicitly counterfactual controls, not ordinary endogenous self-updates.
- Crossed records are separate artifacts, excluded from ordinary forecaster
  training. No original labels/checkpoints are overwritten or silently replaced.

## Validation and execution

Scheduled full CPU suite/lint → compute-node file/checkpoint/batch audit → real
CUDA exact replay and transfer smoke checks in all four streams → crossed array
→ scheduled CPU review and stop. Exact native adapter and per-item replay must
pass before the array starts. Each array task also exactly replays its three
native diagonal updates; a mismatch fails that task rather than substituting
incompatible native observations. Partial cells and failure receipts remain.

Shard by recipient: 24 tasks, each evaluating six off-diagonal branches and
replaying three native updates. Concurrency is capped at 20 allocated GH200
nodes, subject to scheduler/account limits. There are 144 new off-diagonal
branches and 72 identity-replay trainings at most; these remain eight independent
parents. Four extra transfer smoke branches are validation-only and excluded
from scientific analysis.

| Reservation | Maximum additional SUs |
| --- | ---: |
| Full CPU suite, 20 minutes on gg | 0.11 |
| Saved-artifact preparation, 30 minutes on gg | 0.165 |
| Four-stream CUDA replay/transfer validation, 4 hours on gh | 4 |
| 24 recipient tasks × 8 hours on gh | 192 |
| CPU review, 30 minutes on gg | 0.165 |
| **Maximum new reservation** | **196.44** |

These are limits, not predictions of runtime or billed consumption. Queue delay
and partial allocations are additional. Every reservation is persisted before
submission in the existing additional-1,200-SU ledger; completed failures,
minimum charges, replay/smoke work and retries count. Do not reset that ledger or
automatically release reservations for missing accounting. Later confirmation
and policy allowances remain unsubmitted.

Training, evaluation, preparation and statistics run only in Slurm compute
allocations. Login handles staging, hashes and scheduler/account queries. New
GPU submissions require the saved-artifact audit and exact replay dependencies;
the review never releases another experiment automatically. A mechanism result
would require a separately frozen, adequately sized fresh confirmation before
becoming the headline or motivating a new learned selector.

## Current execution status

Implementation is complete. Six targeted synthetic controls passed locally,
including held-out-draw scoring, batch verification, failed-replay rollback and
terminal identity. Repository lint passed. The local Windows pytest temporary-
directory fixture hit a permissions error; its two affected controls passed via
ordinary workspace directories. The full unchanged pytest suite is a required
scheduled CPU dependency, not claimed passed from these local checks.

At approximately 11:54 IST, October 15, Vista accepted the full chain:

| Job | Purpose | Dependency |
| --- | --- | --- |
| 1073234 | Full CPU validation and lint | None |
| 1073235 | Saved checkpoint/batch and scratch-space audit | afterok:1073234 |
| 1073236 | Four-stream exact CUDA replay and transfer smoke checks | afterok:1073235 |
| 1073237_0–23 | Crossed recipient tasks, maximum 20 concurrently | afterok:1073236 |
| 1073238 | CPU review and stop | afterany:1073237 |

The first dependency check showed validation running on an allocated CPU node.
`scontrol` confirmed the GPU array's unfulfilled afterok dependency on1073236,
20-task throttle, eight-hour task limit and disabled automatic requeue. Neither
the artifact audit nor CUDA verification is claimed passed at this point.

Execution source SHA:
`7cb22a077df468347452b9b32e1c4f9756f939ca36b21f8830023166a4b8e7f5`.
Campaign: `/scratch/11617/sujato_ts/third_eye_research/runs/crossed_corrections_7cb22a077df4`.
Local scheduler receipt: `runs/deployment/crossed_7cb22a077df4_submissions.json`.
The source package preserves the declaration as it stood before submission;
this later status update does not rewrite that execution snapshot.

The live account display reported2,031 available SUs. The persistent additional-
work ledger records258.36694SUs charged plus reserved, including this196.44-SU
chain. Reservations are upper bounds, not consumption. The1,200-SU hard cap and
automatic stop after review remain in force. Checkpoints stay on scratch;
the existing work archive is preserved.

At11:55IST, validation1073234 wrote a source-bound passed receipt: **156 tests
passed in54.67seconds and lint passed**. Slurm was still cleaning up the CPU
allocation (COMPLETING); preparation retains its afterok dependency and has not
been bypassed. CUDA and crossed scientific outcomes remain pending.

## October 16 replay failure and bounded diagnosis

The authenticated10:22IST check found1073236 failed after184seconds. The
native diagonal replay produced a different adapter hash. The original versus
replayed first-step losses also differ, so the mismatch is not merely checkpoint
serialization. All24 experiment elements were cancelled before allocation;
1073238 recorded an incomplete review with no crossed states. Original evidence,
logs and exact-replay requirements remain unchanged.

The saved-artifact audit passed for all eight parents, with zero terminal donor
batches. Saved model revisions/configuration identities match. The training and
tokenization source file hashes match the original release, and recorded package
versions match installed metadata. These checks narrow the diagnosis but do not
identify its cause or prove the base model/runtime state is identical.

Next is validation-only instrumentation, not a scientific amendment: full CPU
validation followed by two one-hour GPU diagnostic slots, one per backbone,
maximum2.11SUs including CPU validation. Each restores the same M1 and runs one
discarded training step in four contexts: fresh, repeated fresh, after one-token
generation, and after a loss forward pass. It records input-token hashes,
nonpersistent buffers, first-step losses and actual base-weight checksums.
These probes cannot supply H2 labels or release the main array automatically.
No numerical tolerance is substituted for exact historical replay.

Vista accepted diagnostic CPU validation1075221 and two-element diagnostic
array1075222, withafterok:1075221. Source SHA:
`7f575e13178dccfce8b7bac948e495c74e9efb9729e3bdd6f95522ae5f86606b`.
Output: `/scratch/11617/sujato_ts/third_eye_research/runs/replay_diagnostic_7f575e13178d`.
There is no automatic crossed-array follow-up.

Expanded Slurm accounting confirms every1073237 element was cancelled with
zero allocated nodes and zero runtime. Under the existing declared accounting
rates/minimums, the complete failed crossed chain used0.4975SUs, including CPU
checks/review and the failed GPU preflight. Reconciled earlier CPU diagnostics
are also retained. Total additional charged plus reserved is64.42444SUs after
reserving this2.11-SU diagnosis; unused old crossed reservations are released
only after that complete accounting. The current account display reports2,027
available SUs, which can lag job accounting. Neither number is a science result.

The saved Llama Jinja template contains an unpinned `strftime_now` date in the
system prefix. The runtime host timezone is America/Chicago. Original training
timestamps and a later replay can therefore produce different model tokens from
the same raw examples; the raw manifest hashes do not detect this. This is a
specific hypothesis, not yet a proven explanation of the weight mismatch.

Before any GPU allocation, the unstarted1075222 diagnostic array was cancelled
and preserved at zero GPU cost. Its CPU validation passed156 tests and lint.
The replacement diagnostic retains fresh/repeated-fresh one-step probes, then
replays all50original optimizer steps with the original local calendar date
reconstructed from untouched training logs and recorded elapsed time. Ambiguous
midnight windows are rejected. It compares against the exact original adapter
hash, without using a tolerance or relabeling scientific observations. The
replacement uses two one-hour gh-dev slots plus full CPU validation (2.11SUs
maximum). The main crossed experiment remains stopped pending diagnosis.
