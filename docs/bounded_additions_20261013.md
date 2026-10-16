# Bounded additions to Third Eye

Authorized October13,2026: target about1,000 additional SUs; hard cap1,200.
Historical consumption is excluded from this new envelope. Failed jobs, retries,
CPU charges and Slurm minimum charges are included. No automatic expansion or
8B experiment is authorized by this plan. Existing scientific records remain intact.

## Spending envelope

| Stage | Maximum reservation, SUs | Release condition |
|---|---:|---|
| Paired candidate-LR mechanism: eight development parents, one24h node each |192|Full CPU suite, preparation audit, real CUDA verification|
| Balanced additional small-model states: up to16 nodes ×24h |384|First-stage evidence review and adequate sample/precision plan|
| Recursive practical-policy comparison: up to16 nodes ×24h |384|Freeze rule and independent trajectory plan after development|
| Validation, analysis and preflight overhead |12|Each scheduled job recorded before submission|
| Reserved failure/contingency headroom |228|Explicitly accounted retries; reduce later scope if necessary|
| **Hard maximum** |**1,200**|Never submit beyond charged+reserved headroom|

The first three stages plus overhead reserve972SUs. These are upper limits,
not a mandate to spend all resources. Later stages are protected planning
allowances, not queued experiments; remaining data coverage and measured costs
may require reallocating the envelope before any submission. More outcomes from
the same parent do not increase the number of independent trajectories. A
third backbone is deferred within this cap unless a later review justifies
replacing other work. This envelope cannot guarantee an oral-level result.

## First-stage declaration, before new intervention outcomes

Use all eight completed seed9142/9143 development pilot parents, balanced across
Qwen3-4B/Llama3.2-3B and math/code. These parents were inspected previously;
this is developmental mechanism evidence, never an untouched final test.
Reuse their exact parent checkpoints, verified K=3 correction batches and
candidate optimizer seeds. Intervention changes **candidate learning rate only**
from1e-4 to5e-5. Rank16,50steps, optimizer, microbatch, accumulation, examples,
evaluation and all A2 harvesting/terminal semantics remain fixed. Measure adapter
delta norms as a manipulation check; do not assume they halve with learning rate.
This tests a training hyperparameter, not a pure intervention on effective weight norm.

Continue each intervention M1 three times using existing offsets50M,100M,150M.
Continuation LR remains1e-4 and budgets remain50steps. Original controls use
those same saved draws. Endogenous harvested corrections may change across
the two M1s: the comparison estimates the total effect of the candidate LR,
not a direct effect with continuation data held fixed. Scarcity stays terminal
M2=M1,0steps; report all failures and terminal frequencies without replacement.
There is no new terminal-frequency exclusion threshold.

Split evaluation items by deterministic hash of a declared namespace, role,
item ID and prompt hash. Average both item-fold orientations within each parent.
Lookahead selects using50M/100M on one half and scores150M on the other.
Immediate selection uses M1 on the selection half. Never select a future maximum
and score it on the same outcomes. Fractionally average tied maxima at12decimals.
The primary mechanism contrast holds the baseline-immediate selected candidate
fixed: half-LR minus baseline future utility on the scoring draw/items. Retention,
reselection and all-draw averages are secondary. A simple practical rule is
declared now: among candidates with positive equal-head immediate utility and
nonnegative retention change, select maximum immediate utility; otherwise hold.
Compare it with forced immediate selection and the unchanged parent. This rule
is a development diagnostic until independently replicated through T=5.
The main online implementation now supports `guarded_h1`. It evaluates the same
three M1 branches as immediate greedy selection, uses a fixed seeded uniform
choice over eligible tied maxima, and commits the saved chosen M1 without a
fourth update. Holding preserves the parent exactly. The development item-fold
analysis averages tied choices; historical greedy tie behavior is preserved.
This costs no H2 harvesting at deployment; it still incurs K=3 immediate trials.

Report paired trajectory-level intervals/sign-flip tests and Holm adjustment
across the displayed comparison family; eight parents, shared fixed item pools and earlier
inspection limit inference. Folds, candidates and continuation draws are not
extra independent observations. Effect sizes remain signed, with no oracle
regret clipping. No sample-size claim or superiority decision follows from
passing implementation checks.

## Execution and stopping

Full CPU suite → compute-node input/statistics audit → two-backbone real CUDA
half-LR/rollback verification → eight24h GPU tasks, at most eight simultaneous
nodes → CPU review after all tasks terminate. Failed dependencies block launch.
Each reservation is persisted before sbatch, including ambiguous submission
attempts. Never retry automatically or release reservations on a failed state
without complete Slurm accounting. Login nodes perform only source staging,
standard-library metadata checks and scheduler operations. The final review
stops; no controller releases the later384+384SU allowances automatically.

Measured prior five-draw pilot collection averaged13.58h per state. Three-draw
fixed-batch interventions should cost less, but24h per-task limits protect the
cap. Queue delays are outside these limits. Partial outputs remain diagnostic
if a task reaches its limit. Future recursive comparisons must include no-update,
immediate selection and the frozen practical rule, balanced across both task
families and backbones. Learned H2/H1 comparisons and learning curves remain
eligible only if data and held-out precision justify them within the same cap.

## Submission receipt

On October13, Vista accepted this first-stage chain:

| Job | Purpose | Dependency | Maximum SUs |
|---|---|---|---:|
|1068784|Full CPU suite and lint|None|0.11|
|1068785|Input audit and independent seed/item development analysis|afterok1068784|0.165|
|1068786|Real two-backbone half-LR and adapter rollback verification|afterok1068785|2|
|1068787_0–7|Eight paired intervention parents, up to eight concurrent GPUs|afterok1068786|192|
|1068788|Review complete or partial evidence; stop|afterany1068787|0.165|

Total reservation **194.44SUs**. Immutable main execution source:
`421fb73bac2c9290ea8cb2e9e0cc1fc2262128c9cb1caec08e88ed52e76ec31c`.
Campaign: `/scratch/11617/sujato_ts/third_eye_research/runs/bounded_additions_421fb73bac2c`.
Budget ledger: `/scratch/11617/sujato_ts/third_eye_research/runs/bounded_additions_20261013_budget.json`.
The live account query before submission reported2,093SUs. Later stages are
not queued and this receipt establishes no scientific outcome.

Full scheduled CPU validation1068784 subsequently passed **147 tests in51.82s**
and lint, with a source-bound `passed=true` receipt. The GPU dependencies remain
in force; full-suite success is an implementation check, not a research result.

The input audit1068785 also passed. Its prespecified independent seed/item
development comparison on the reused eight parents estimates future-selection
minus immediate-selection utility at−1.62percentage points, with Holm p0.469.
The guarded rule's advantage over forced immediate selection is+0.68points,
CI[−0.70,+2.33], and its utility relative to holding is−0.29points,
CI[−2.56,+1.45]. These are development diagnostics with no established useful
forecasting or guarded-policy advantage. They do not justify releasing larger
collections. The paired LR test remains the next controlled scientific question.
At11:12IST, CUDA1068786 was pending for priority; mechanism and review jobs
remained behind their declared dependencies.

## Provenance correction before GPU execution

An export defect was found before GPU allocation: copied native control fields
included the old H2 label/checkpoint alongside the intervention's actual
replicates. The comparison code used actual replicates, but the export was
ambiguous. The corrected main implementation removes inherited H2 outcomes,
old continuation availability/hash and baseline-LR feature probes; it saves
measured H1 and explicit continuation replicates only. A dedicated synthetic
test checks preservation of the native input and exclusion of stale fields.
Candidate-LR, continuation, scoring and budget semantics are unchanged.

Jobs1068786/1068787/1068788 were cancelled before allocation. Complete Slurm
accounting, including all eight cancelled array elements, confirms0GPU SUs.
The completed CPU jobs cost0.165SUs combined under minimum charging. Their source,
preparation outputs and submission/cancellation receipts remain preserved.

Replacement main source:
`ce71b7956a1806a086838bbb89d4554d0043428f863513c0c1239a0fb2e9c624`.

| Job | Purpose | Dependency |
|---|---|---|
|1068800|Full CPU suite and lint|None|
|1068801|Input audit and independent seed/item analysis|afterok1068800|
|1068802|Real two-backbone half-LR/rollback verification|afterok1068801|
|1068803_0–7|Eight paired intervention parents|afterok1068802|
|1068804|CPU review and stop|afterany1068803|

New campaign: `/scratch/11617/sujato_ts/third_eye_research/runs/bounded_additions_ce71b7956a18`.
The same persistent budget ledger includes both executions. Charged old checks
plus new reservations total194.605SUs; later-stage protected allowances remain
unsubmitted. The total additional cap stays1,200SUs.

Replacement scheduled validation1068800 passed **148 tests in57.33s** and lint
with the exact replacement source hash. At11:19IST it was leaving COMPLETING;
the preparation/CUDA/GPU/review chain retained its dependencies.

## Live progress: October14,11:28IST

CPU validation, preparation and real CUDA checks completed successfully. All
eight1068803 array elements are running on allocated GPU nodes, each with
elapsed16,257s (4h30m57s) at the check. No task failure receipt was present.
Completed intervention candidate bundles by index0–7:2,2,1,1,2,2,1,1
(12of24). Saved continuation observations:8,6,5,5,6,6,3,3 (42of72).
These are artifact counts, not a percentage of overall research completion or
a scientific outcome. No complete-parent receipt exists yet. Review1068804
remains queued behind the array and will start after all elements terminate.

The running array has accrued about36.13GPU-node hours at the snapshot.
The account display still shows2,093 available SUs; active allocations may not
yet be reflected there. The immutable first-wave reservation and overall
1,200 additional-SU cap remain in force. No later confirmation jobs were released.

Later update: by October14,22:25IST all eight jobs and the review completed.
The primary half-LR selected-candidate effect is−0.24pp, CI[−0.97,+0.37],
with no demonstrated benefit. All-candidate mean effect+0.98pp is secondary
and not significant after adjustment. Total first-wave use61.65194SUs; native
results, accounting and the small CPU-only follow-up are recorded in
[the completed mechanism review](mechanism_review_20261014.md). Later GPU
allowances remain held; there is no automatic confirmation release.
