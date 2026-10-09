# SmolLM3 external confirmation: execution record

Authorized October8. Follow [S1](protocol_smollm_supplement_v1.md): four single-state
experiments, two seeds6142/6143 for code and mathematics, with unchanged A2
budgets and K=3/H=2 labels. This is separate from the fixed E2 primary cohort.
It does not support a completed T5 third-backbone or third-backbone noise-floor claim.

## Current state

Verified October 9, 2026, approximately 09:30 IST. The setup receipt reports
`status=passed` and binds the prepared tasks to the frozen release. Official
model revision: `a07cc9a04f16550a088caea529712d1d335b0ac1`.

| Stage | Job/status | Evidence |
| --- | --- | --- |
| CPU suite, tokenizer contracts, official weight staging, task preparation | 1057831 completed in88s | All127 checks passed; official public weights downloaded; revision and file hashes pinned; math/code masking contracts passed |
| Both-task CUDA/LoRA/probe/checkpoint/forecast-schema verification | 1057855 completed in 8m43s | Both tasks passed real 50-step training, rollback, checkpoint and forecast-schema checks |
| Four single-state collections | 1058281, 1059789, 1059790, 1059791 queued | Both-task validation passed; one state per invocation, all checkpoints retained |
| Fixed forecast transfer/selection review and archive | 1059792 queued after all four collections | No fitting or method selection; incomplete coverage stops full comparison |

Setup resolved the official repository before scientific outcomes. The model
uses the same pinned runtime as the core study. Gold training references are
compatibility fixtures only, with their adapters discarded before collection.
Scientific updates use verified self-generated corrections exclusively.

Submission records are under
`runs/a2/empirical_v1/smollm_supplement_v1/submissions.json`. The dispatcher performed scheduler
and small-metadata operations only on the login node; model download, tests,
training, harvesting, inference, evaluation and analysis use Slurm compute nodes.
It counts all queued/running GPU nodes plus future core control reservations
before submitting optional work. It does not allocate a GPU to wait for capacity.

Submission cutoff is fourteen hours before October11 23:59:59 IST; each GPU
process also enforces the experiment deadline. Source/scope/task/model identities
are immutable, failed states are preserved and no automatic replacement is allowed.
Current scientific collection is pending GPU allocation, not completed.

The initial dispatcher stopped when submission of index 1 was rejected with
`Job dependency problem`: its successful validation job had left Slurm's
dependency lookup. Only index 0 had been accepted. A separate scheduler-only
handoff preserved the stopped record, checked complete accounting for unexpected
jobs, required successful validation accounting and matching release/task
receipts, and queued only the remaining three indices. It omitted the already
satisfied validation dependency; frozen runtime receipt checks still apply.
Three local handoff checks passed. No protocol or scientific budget changed.
Handoff source, accounting, receipts and the original stopped log are preserved
inside the supplement and will be included in its final archive. All four jobs
and their review are now submitted; no waiting dispatcher is needed.

## Core queue protection

The core launcher-path repair was necessary before collection: a generated
filename still referred to V2 while the validated relay required V3. The frozen
V3 code is preserved; a versioned operational wrapper changes only that filename.
All147 native and wrapper CPU checks passed in job1057803. Original all-four-stream
CUDA verification1057730 completed in15m51s. No candidate or optimization budget
changed, and pending entries were retired before any experimental execution.

The replacement core collection jobs are1057804–1057811, with confirmation1057812.
Their submission receipts are under `checkpoint_fallback_v3/path_repair_v1/`.
The S1 dispatcher reads this current graph when reserving its eight future core
controls. After the three recoveries finished, all four SmolLM experiment
entries fit alongside eight core trajectories and eight future core controls:
twelve queued GPU nodes plus eight reserved controls equals the twenty-node cap.

The scope is a modest expansion from two to three backbones if validation and
four-state coverage succeed. Keep the existing primary comparison unchanged;
evaluate only the already frozen Direct-H2seed42 and matched-H1seed42 transfer
and observed selection values. Interpret four early states with limited power.
