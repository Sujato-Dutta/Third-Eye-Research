# Final empirical cohort: execution record

Submitted October22 under the [final declaration](final_confirmation_20261022.md).
No forecaster fitting or forecasting comparison is part of this cohort. H2 is
an observed continuation outcome. Historical forecasting nulls remain intact.

Immutable source:
`43b5c215c6b089841af7db31ad4a0f783afbc1f105b4c978808e278f0024ea06`.
Archive SHA256:
`a618c53d9df25b19520dbba09199ea279f135df7687828fa30ec1d5bb989b59a`.
Remote campaign:
`/scratch/11617/sujato_ts/third_eye_research/runs/final_empirical_confirmation_43b5c215c6b0`.

## Submitted jobs

| Job | Work | Dependency | Maximum SUs |
| --- | --- | --- | ---: |
| 1089033 | Full CPU tests/lint | None | 0.110 |
| 1089034 | Freeze seeds/items; precision/provenance checks | afterok:1089033 | 0.165 |
| 1089035 | Direct code cost/verifier check, development GPU | afterok:1089034 | 0.500 |
| 1089036, elements0–7 | Llama code, at most5 GPUs | afterok:1089035 | 114.000 |
| 1089037, elements0–7 | Llama math, at most5 GPUs | afterok:1089035 | 146.000 |
| 1089038, elements0–7 | Qwen code, at most5 GPUs | afterok:1089035 | 92.000 |
| 1089039, elements0–7 | Qwen math, at most5 GPUs | afterok:1089035 | 156.000 |
| 1089040 | Verify cohort; write statistics/tables/plots | afterok:all arrays | 0.330 |

32 independent initial parents, eight per stream; K=3, size-two corrections,
three native continuation draws, unchanged A2 budgets/rendering. Scoring is
128 target/128 OOD/256 retention items per task, independently frozen and
excluding the eight code cost-calibration items. Score13 checkpoints per parent.
The single primary is Qwen-minus-Llama future forced-update utility, with
joint parent/shared-item uncertainty. Preserve the prior unsuccessful joint
pooled-harm/interaction design; this narrower primary was declared before data.

All tasks and the results job are submitted, with at most20 concurrent GPUs.
They survive loss of SSH/local folder closure. No local controller is needed to
release remaining tasks. Invalid dependencies cancel downstream jobs; nothing
retries, changes seeds or expands the cohort automatically. Login operations
only stage/hash files, reserve/account the ledger and submit/inspect Slurm.

## Validation and preparation

At **14:10:52 IST**, validation1089033 had COMPLETED, exit0, in87seconds:
**207 tests passed in72.46seconds**, plus lint. Controls exercise32 configuration
and item freezes, immutable file checks, non-forecasting actions,13 checkpoint
references, shared-item uncertainty and complete synthetic report/plot creation.

Preparation1089034 COMPLETED, exit0, in85seconds. The exact single-primary
projection gives **85.65% conditional planned power**, using a5.758pp shrunk
pilot gap and1.912pp inflated SD. This is planning, not confirmed real-data power.
Seed collision, suite integrity and shared-pool checks passed. Code check1089035
was RUNNING on `c642-031`; the four research arrays awaited its completion.
No fresh scientific outcome was inspected or reported at that check.

## Budget and final deliverables

Complete maximum reservation: **509.105 SUs**, within C's560-SU ceiling. Earlier
design jobs were reconciled to0.165SU. The bounded ledger is **742.6767 SUs
used/reserved**, leaving **457.3233 SUs** under1,200. Reconcile final terminal
allocations/minimum billing afterward; historical original campaigns are separate.

The final job requires all32 complete parents and writes `evidence/report.json`,
`policies.csv`, `findings.md`, `policy_utility.pdf`, `policy_utility.png` and
completion/provenance receipts. It stops at
`EXPERIMENTS_COMPLETE_FREEZE_EVIDENCE_AND_WRITE` whether the hypothesis replicates
or not. No further architecture, forecasting recovery, model, recursive study
or favorable-seed collection follows automatically.

Receipt: `runs/deployment/final_empirical_confirmation_43b5c215c6b0_submissions.json`.

## Runtime correction and replacement scheduler jobs

The status and reservations above describe the original submission. Direct code
timing1089035 stopped at its cost bound: HumanEval scoring exceeded the MBPP
proxy. All32 original parent tasks and review were cancelled before allocation;
their failure/cancellation records remain intact. No fresh research outcome was
used to change the experiment. See the
[runtime-only declaration](final_confirmation_runtime_correction_20261022.md).

The unchanged32 seeds, items, actions and scientific source were resubmitted:

| Job | Work | Dependency |
| --- | --- | --- |
| 1089044 | Extended-source CPU validation | None |
| 1089045 | Complete direct code cost verification | afterok:1089044 |
| 1089046, elements0–7 | Llama code, at most5 GPUs | afterok:1089045 |
| 1089047, elements0–7 | Llama math, at most5 GPUs | afterok:1089045 |
| 1089048, elements0–7 | Qwen code, at most5 GPUs | afterok:1089045 |
| 1089049, elements0–7 | Qwen math, at most5 GPUs | afterok:1089045 |
| 1089050 | Final statistics, tables, plots and findings | afterok:all arrays |

Validation1089044 completed exit0: **209 tests passed**, plus lint. Code
verification1089045 wrote a passing readiness receipt and was COMPLETING at the
latest check. Dependency release waits for Slurm completion; it is not bypassed.
The code timing report hash is
`6ef5e0ea7ca08b2463f532e00966b79ab70d01c421d498d048aca9b32ab1a98f`.
The verifier source is `4c0101a1064b`; actual scientific collection and final
review retain source `43b5c215c6b0`. All original scientific files are unchanged.

Only Qwen code runtime increases, from11.5 to14.5hours per parent. Maximum
complete stage cost, including allocated original checks, is **533.355 SUs**,
below560. The bounded ledger is **766.9267 SUs used/reserved**, leaving
**433.0733 SUs** under1,200. All32 tasks and final reporting are submitted;
up to20 GPUs may run concurrently. No forecasters are fitted or compared.

Replacement receipt:
`runs/deployment/confirmation_runtime_4c0101a1064b_submissions.json`.

## October23 live progress

At09:37:38 IST, all20 first-wave GPU jobs were RUNNING, five per stream,
with04:22:28 allocated runtime each. All12 remaining parents were PENDING on
the declared array concurrency limits. Final review1089050 remained PENDING
on successful completion of all four arrays. Both validation1089044 and code
cost1089045 were COMPLETED, exit0. There were no failed cohort jobs.

Metadata and bounded log-tail checks confirmed20 collection directories, zero
complete parents and zero final-scoring progress receipts. The jobs were still
collecting/evaluating candidate continuations. No traceback, CUDA out-of-memory
or Slurm error appeared in the20 inspected log tails. This check does not
estimate scientific effects or change any rule, seed, item or budget.

Raw scheduler and log-metadata receipts:
`runs/deployment/vista_crossed_requests/189-final-cohort-progress.response.json`
and `190-final-cohort-log-metadata.response.json`.

At14:29:49 IST, **14/32 parents completed successfully**: Llama code5/8,
Llama math4/8, Qwen code5/8 and Qwen math0/8. Each completed parent has13 final
checkpoint score receipts, a passing completion record, matching scientific
source and zero forecaster fits. Fifteen jobs were RUNNING and three Qwen math
jobs awaited the array throttle. No cohort job failed; inspected log tails
contained no traceback, CUDA memory or Slurm error. No effect estimates were
inspected to make a scientific decision.

With only18 original parents unfinished, Qwen math's scheduler concurrency
limit was raised from5 to8. This releases its last3 tasks into spare capacity
while keeping total possible concurrent work below20. Seeds, data, actions,
implementation, update budgets and SU reservations are unchanged. The final
analysis still depends on all32 parents. Scheduler change and metadata receipts
are `191-final-cohort-progress.response.json`,
`192-final-cohort-completion-metadata.response.json` and
`193-final-cohort-spare-concurrency.response.json` in the request archive.
At14:31:56 IST, Slurm confirmed all3 released tasks RUNNING: **all18 remaining
parents were running, with no research task left pending**. Final review alone
awaits their completion (`194-final-cohort-released-status.response.json`).
