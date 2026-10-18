# C1: Explicit temporal context in model inputs

Declared October16,2026 before collecting any corrected crossed-matrix outcomes.
This corrects an implementation-level input-reproducibility defect; it does not
change A2 correction harvesting, candidate size, optimizer budgets or terminal
semantics. Original source snapshots, labels and failures remain immutable.

## Evidence and unresolved confirmation

The saved Llama chat template inserts `strftime_now("%d %b %Y")` into its system
prefix when `date_string` is absent. Original configs did not supply that field.
The backend and tokenizer source hashes and recorded package versions match,
yet the failed replay has a different first-step loss and final adapter hash.
Thus raw example hashes and random seeds alone did not freeze the model inputs.
The current evidence identifies a concrete date-dependent input difference;
the exact contribution to the weight mismatch still requires the queued test.

Diagnostic1075226 restores the original process-local date from original training
logs and replays all50steps. Its CPU validation1075225 passed156 tests and lint.
The original runtime uses America/Chicago; UTC dates cannot simply be substituted.
Training windows spanning local midnight are rejected as ambiguous. The corrected
matrix's preparation must require exact restored-date adapter matches for both
backbones, and check available original base-weight checksums. A diagnostic
completion flag alone does not satisfy this prerequisite.

## Rendering repair

All prompt, SFT, generation, likelihood and diagnostic-forward entry points now
reject an implicit date for a date-dependent template. An explicit runtime
`prompt_context` is merged with configured chat kwargs; conflicting dates are
rejected. Checkpoints and per-item evaluation artifacts record the effective
kwargs and template hash. Reloading a new checkpoint with a different recorded
context fails. Legacy checkpoints are identifiable by their missing context;
their metadata is not retroactively rewritten.

The new diagnostic matrix uses the fixed context `date_string="11 Oct 2026"`
for every new update and evaluation, including Qwen where the kwarg is unused.
This value is declared before new outcomes and is not selected for accuracy.
Original saved M1 weights and verified correction texts remain the inputs.
Their historical generation/training context is part of their provenance; this
is not a fresh end-to-end cohort with fully pinned histories.

## Do not mix historical and corrected scores

Historical diagonal scores are not used to fill the new matrix. Retrain and
evaluate **all216 recipient × donor × draw cells**, including72 fresh native
diagonals, under the common fixed context. Re-evaluate parent and M1 controls
under that same context. The primary native-pairing contrast and all secondary
comparisons are unchanged mathematically; every term now belongs to the same
declared rendering condition. Historical H1/H2 labels remain historical.

In addition, retain72 original-date, exact-adapter identity replays as separate
validation-only records. Each recipient task performs its three original
replays plus nine fresh matrix updates and two baseline evaluations. Failed
replays still stop the task. Old evaluation dates may differ, so the four-stream
CUDA preflight tests repeated full evaluations under the fixed context instead
of equating historical scores to newly rendered inputs. Transferred empty
donor batches preserve the recipient exactly and use its newly scored M1.

The new input context must be present in every matrix/baseline artifact. The
review rejects historical or differently rendered artifacts, changed batch
identities, budgets, seeds and checkpoints. Baseline parent evaluations repeated
across recipients must agree exactly. Fresh diagonal weights are not required
to equal historical weights trained on another rendered date; the separate
original-context replay must equal them.

## Deployment boundaries and budget

New full CPU validation → preparation after both CPU validation and1075226 →
four-stream CUDA replay/repeated-evaluation check →24recipient tasks, up to20
concurrent GH200 nodes →CPU review and stop. Preparation also validates saved
inputs, original date evidence and scratch capacity. No GPU matrix is eligible
merely because this correction was implemented.

The replacement retains the196.44-SU maximum chain reservation: CPU checks,
four-hour CUDA verification and24eight-hour recipient slots. Additional native
scoring and replay work count inside those task limits; this is not a wall-time
prediction. The same persistent1,200-SU ledger includes old failures, cancelled
jobs, diagnostics and retries. Cancelled unallocated tasks cost zero only after
complete accounting. No further forecasting, scaling or confirmation launches
automatically after review.

Historical Llama comparisons across different days may have an additional
rendering-context confound. Their exact scope and size are not yet quantified;
do not infer that this explains the forecasting results or rescues the paper.
Qwen's inspected template has no corresponding dynamic-date field. Any later
paper must disclose this issue and use a properly frozen fresh cohort for
confirmatory claims, rather than silently pooling old and corrected outcomes.

## Confirmed diagnosis and execution status, October16 evening

At approximately20:52IST, the live check confirmed both1075226 diagnostic
elements completed successfully. With the original Llama date restored to
11Oct2026, all50steps reproduced the original adapter exactly:
`009c29292a1b4ce6d4b47eb5dd6b690c7c860477e215939fbaf2b0ff723d1993`.
The first-step loss also matched exactly. Fresh/repeated-fresh executions on
the later date agreed with each other but had different encoded inputs and
did not match the historical update. Original Llama base-weight file checksums
matched. Qwen's negative control reproduced its full original adapter, and its
encoded inputs were unchanged by the date kwarg. This confirms the date defect
for the investigated replay; it does not quantify effects on prior paper claims.

Corrected-source CPU validation1075255 passed **162 tests in57.03seconds** and
lint. Preparation1075256 passed the date-proof and all-eight-parent input audit.
The immutable corrected source is
`a274ff920ac613d9ec78dad280e4dd153ea345bec1727cfbea2efbb929ab60df`.
Campaign: `/scratch/11617/sujato_ts/third_eye_research/runs/crossed_corrections_a274ff920ac6`.
Retrieved proof/audit artifacts: `runs/deployment/evidence/rendering_c1_20261016`.

Four-stream CUDA validation1075257 is pending allocation. Its pending request
was moved togh-dev with a two-hour cap for backfill; the original conservative
4-SU reservation remains held. Both queues were fully allocated or in cleanup
at inspection, so there is no reliable start estimate. Matrix1075258_0–23 remains
dependent on successful CUDA validation, capped at20concurrent nodes; review
1075259 follows the matrix and stops. No scientific matrix cell is yet complete.

Completed diagnostic/CPU accounting has been reconciled. Additional charged
plus reserved is259.30944SUs under the1,200-SU cap, including the still-unstarted
matrix's192-SU upper reservation. The account display shows2,026available SUs.
This later status update does not alter the pre-outcome declaration saved in
the immutable execution release.

At20:58:50IST on October16, the scheduler confirmed1075257 RUNNING on
`c642-042`, started20:55:50IST, with180seconds elapsed and a two-hour limit.
The24-task matrix and final review remained pending their dependencies.
This is a validation execution milestone; no new matrix result was available.
