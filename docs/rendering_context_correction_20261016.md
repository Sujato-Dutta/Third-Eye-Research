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
