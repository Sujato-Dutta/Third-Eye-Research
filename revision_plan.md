# Third Eye: review-driven revision implementation plan

**October29 evening:** all16 [matched controls and their analysis](docs/dgx_controls_results_20261029.md)
completed. Shorter-step, pooling and recipient-state contrasts remain
inconclusive. A reference-solution source effect is significant but confounded
with text style and label-token exposure; the completed read-only audit
documents these differences without establishing mediation. Storage rotation
and Qwen8B GPU validation passed; its first parent is running, followed by
the remaining Qwen parents and the queued Llama8B block. No additional
experimental arm or Vista GPU job was introduced by the audit.

**October29 progress:** the16-parent SmolLM extension and final analysis are
complete; [results](docs/smollm16_results_20261029.md) support further continuation
loss, with harm already visible at H1. DGX controls are15/16 complete, with
the final element running and both8B blocks queued afterward. Full accounting
reduces actual completed Vista revision usage to218.1069 SUs, leaving281.8931
under the500-additional-SU cap. Earlier reservation figures remain historical;
no new Vista GPU job has been submitted with the released headroom.

**October28 scale authorization:** queue Qwen3-8B and Llama3.1-8B-Instruct
behind the complete DGX controls and their analysis. Each adds eight fresh
parents (four math/four code), with unchanged training, harvesting and scoring.
This is a separately frozen scale-sensitivity extension, not an equal-N
replacement for the main cohorts. The serial dependency and lossless storage
handoff are specified in [the8B execution record](docs/dgx_scale_execution_20261028.md).
It consumes no Vista SUs; the500-additional-SU ceiling is unchanged.
Both blocks are submitted: Qwen array29452, Llama array29456, with serial
preparation, compatibility, statistics and storage dependencies after28974.
The full source passed230 CPU tests and lint in29448. The original2--3-day
estimate below predates this serial8B extension and does not cover its runtime.

October24,2026. **SmolLM3 sixteen-run extension approved for execution.**
The matched continuation and recipe controls are now authorized for DGX;
[their separate execution record](docs/dgx_revision_execution_20261026.md)
documents the one-GPU chain. They are not charged to the Vista SU allowance.
**October26 operational update:** both SmolLM GPU checks passed. Measured
complete costs exceed the original14h limits. The same16-run study now reserves
23h per code parent and30h per math parent; all scientific settings remain
unchanged. Other GPU-control blocks are deferred on Vista under the500-SU cap
and have instead been submitted on DGX, pending verified input transfer.
**October26 allocation instruction:** use Vista first within this allowance;
DGX takes work that cannot fit. Preserve the sixteen queued SmolLM parents on
Vista and retain the remaining Vista allowance for finishing work and complete,
measured follow-ups. Transfer only the frozen inputs needed by DGX controls,
not the Vista project or model cache. This is an allocation change, not a new
scientific protocol or a duplicate launch of the same tasks on both clusters.
The latest allowance is **500 additional SUs from now**, superseding the earlier
800-total interpretation. Existing completed results, declarations and manuscript
remain preserved. No forecasting development is included.

## Objective and priorities

Determine whether the immediate--continued contrast reflects candidate selection,
the continuation recipe, the receiving model state, the generated training data,
or a combination. Strengthen model coverage with a balanced third-family study.
Conclusions and title must follow the controls; a successful new method or a
particular publication outcome is not an execution criterion.

The first review is right to request role-level results and clearer prose. The
second identifies the most important scientific control: the first update was
selected, while its continuation was not. Correcting that asymmetry precedes
claims about hidden fragility. A near-zero H1 mean is not a safety finding.

The reported -5.10pp is the **sum**, not the mean, of target and OOD changes:
`3 * aggregate - retention`. Their combined two-role mean is about-2.55pp.
Neither value identifies the separate roles or establishes catastrophic collapse.

## Budget and verified resources

Vista reported **1,531 available SUs** for allocation IRI23021 at11:34:41 IST,
October24; its expiry is April30,2027. No job was queued or running. This is
the service-reported account balance, distinct from our project ledger.

The bounded-work ledger is542.573611 SUs used/reserved. The proposed additional
ceiling gives at most**1,042.573611 SUs** in that ledger, still below its existing
1,200-SU cap. On October26 the live ledger's hard cap was reduced to
**1,042.573611 SUs**, enforcing the500-additional-SU ceiling directly; its
previous contents were preserved. Earlier original campaigns are excluded
from this ledger. Account balance does not increase this authorization.

| Work | Fixed scope | Maximum additional SUs |
| --- | --- | ---: |
| Saved-data analyses and full CPU validation | All32 completed parents | 4 |
| Integrated GPU compatibility and cost checks | Existing models plus SmolLM3, both tasks | 6 |
| Matched recipient/data-source controls | Deferred; original16-parent design preserved | 0 |
| Steps, gold-solution and pooling controls | Deferred; original16-parent design preserved | 0 |
| Fresh third-family confirmation | SmolLM3,8 seeds per task;16 parents | 424 |
| Final statistics, artifact checks and manuscript regeneration | One integrated report | 4 |
| Unreserved allowance for retries or a later complete control block | Preserve failures; same seeds | 62 |
| **Hard additional maximum** | **All blocks, including contingency** | **500** |

The initial budget assigned128 SUs to controls,96 to recipe checks and224 to
SmolLM. The complete measured three-draw SmolLM cost instead requires424 SUs
in reservations. Running all three original blocks would exceed500 even
before contingency; only the explicitly approved SmolLM block is submitted.
The other complete control designs below are preserved as future work, rather
than silently reducing their sample sizes or promising them within this budget.
At12:24 UTC October26, including the allocated input export,426.0175 revision
SUs were charged or reserved, leaving73.9825 available on Vista. These are
conservative full-walltime reservations, not426 SUs already consumed. Actual
charges can release headroom. Use this remaining allowance on Vista for
finishing work or complete follow-ups whose measured total fits; account for
all failures and minimum billing before each submission. The full16-parent
paired-control block is overflow work on DGX. Do not split a paired parent
across sites or duplicate a task to spend the remainder. The frozen
[allocation receipt](experiments/revision_20261024/allocation_20261026.json)
records the enforced cap, jobs and bounded transfer scope.
Scientific budgets are unchanged. Operational details and job IDs are in
[the execution record](docs/smollm16_execution_20261024.md).
These are **allocation ceilings, not measured
runtime promises**. Before bulk submission, price each stream using the existing
direct scoring rates, archived harvesting/training times, and one integrated
preflight. Include model loading, generation, sandbox verification, scoring,
minimum billing and at least25% timing margin. A control task's limit covers
every arm in that task, not each arm separately.

If runtime coverage requires longer tasks, transfer available contingency into
their reservations **before** submission and retain the500-SU ceiling. Never
silently shrink scoring sets, reduce repeats, omit an arm, or change seeds to fit.
If the complete design cannot be priced within500, report the exact shortfall
before launching an incomplete scientific design. Account availability does not
authorize spending beyond500. Expected spending may be below the ceilings.

## A. Saved-data analysis: no GPU execution

Run these together in one allocated CPU analysis. They incur small CPU SUs;
calling them literally zero-cost would be inaccurate.

1. **Selection control, all32 parents:** report uniform-candidate G1 and G2,
   the existing development-selected G1 and G2, and their paired continuation
   changes. Average continuation draws within candidate and candidates within
   parent. Separate selected-versus-uniform changes at each horizon. The
   uniform comparison is descriptive of the existing candidate distribution,
   not a randomized test of all possible candidate generators.
2. **Role and stream decomposition:** target, OOD, retention and aggregate at
   H1, H2 and H2-H1; overall and for all four existing streams. Add an
   exploratory model-by-task difference-in-differences with joint uncertainty.
3. **Transitions:** a32-parent G1/G2 scatter; counts and denominators for
   positive-to-negative, negative-to-positive and zero outcomes, both selected
   and uniform. Retain fractional tie weights. Sign changes of noisy observed
   means are descriptive, not established individual latent effects.
4. **Output-format audit:** answer extraction, syntax/test failures where
   distinguishable, output lengths, and recorded generation-limit hits. Compare
   paired H1/H2 outcomes on identical items. Use saved finish reasons/generated
   token counts if present; otherwise label retokenized lengths as proxies,
   not proven truncation. Do not add answer-repair or longer-generation scores
   to the original primary endpoint.
5. **Harvest and exposure audit:** pool size, verified corrections per attempted
   revision, unique-problem counts, token exposure and update times. With
   microbatch1 and accumulation8,50 steps present400 example instances:
   nominally200 passes over2 examples. Ten and25 steps imply40 and100 passes.
   These are example passes, not equal token exposure across text lengths.
6. **Statistical robustness:** retain original run/shared-item bootstrap results;
   add a fixed-seed paired sign-flip test of continuation increments as a
   conditional-on-items check. Explain its symmetry assumption. Do not permute
   model labels as if pretrained models had been randomly assigned or treat
   draw/item observations as independent parent replications.

Keep the original confirmation primary and adjusted secondary family intact.
New analyses are explicitly review-driven and exploratory.

## B. Frozen subset and matched recipient/data-source control

DGX execution uses supplement R-DGX1 in the execution record: re-score both
receivers and repeat the native50-step baseline on DGX for every selected draw.
This prevents direct pairing of an A100 arm with an archived Vista outcome.
The96-cell count below describes missing cells under the original Vista design;
DGX additionally recomputes32 native baselines. All16 parents, draw choices,
correction batches and recipe definitions remain fixed.

Use **16 existing parents**, four per model/task stream, selected by a published
hash ordering of identifiers before reading per-parent effect estimates. Save
their identities in `experiments/revision_20261024/manifest.json`. They are
paired follow-ups on inspected parents, not16 independent fresh replications.

For each parent, use the original H1 selection distribution. Resolve ties with
a separately frozen hash-random draw without using final scores; pair every new
arm to that exact candidate's archived baseline, not a different tie mixture.
Choose two of the three existing continuation draws by fixed hash order.

For each draw, D1 is the saved size-two verified batch generated by M1. Generate
one fresh size-two native D0 batch from M0 with the original harvest budget,
training pool and a disjoint, frozen seed. Run this2x2 design:

| Receiver | D0: M0-generated corrections | D1: M1-generated corrections |
| --- | --- | --- |
| M0 | New native first-update control | New matched-data receiver control |
| M1 | New crossed-data control | Reuse original native continuation |

Train each new cell for50 steps with a fresh AdamW optimizer and matched training
seed within draw. Restore its receiver before training. Score on the unchanged
128/128/256 final items. This creates **96 new scored checkpoints**:
16 parents x2 draws x3 missing cells. No existing M1 or M1/D1 outcome is rerun.

Primary diagnostic contrasts are the difference of update increments between
M1 and M0 on identical D1, the analogous difference on D0, and their interaction.
Subtract each receiver's own pre-update score when measuring its increment;
also show total utility versus the original M0. The native diagonal comparison
answers whether the full model/data transition differs from updating M0.
Data-source differences include problem composition and solution content;
they are not automatically a pure correction-quality effect.

## C. Continuation recipe and reference-solution controls

Reuse exactly the same16 receiving M1 checkpoints, selected D1 batches and
training seeds. The original50-step M1/D1 cell is the paired baseline.

- **10 and25 steps:** two draws per parent and both step counts;64 new models.
  Use the same data and optimizer-seed schedule as the archived50-step update.
  These test continuation intensity, not shortening the initial M0-to-M1 update.
- **Gold solutions:** replace each D1 completion with the dataset's complete
  reference solution for the same problem; two draws,50 steps;32 new models.
  GSM8K uses its reference worked answer, MBPP its reference code. Verify
  references before training and preserve failures. Log differing token lengths;
  this compares solution sources, not equal-token exposure or verification itself.
- **Pooled corrections:** union the chosen candidate's three archived native D1
  batches, deduplicate by problem ID with a frozen tie rule, and train once from
  M1 for50 steps;16 new models. Pool size may be2--6; retain every parent and
  report the realized size. No duplicates, new harvesting or conditional exclusion
  to manufacture a larger pool. This is a continuation-pooling strategy, not a
  clean batch-size-only intervention or a replacement for the original K=3 rule.

This block creates **112 new scored checkpoints**. All recipe arms execute
regardless of whether the saved-data diagnostics favor the current title.
If references or frozen checkpoints are structurally invalid, stop the affected
task with a recorded defect rather than replacing its scientific unit.

## D. Third model: proper family coverage within the budget

**Use HuggingFaceTB/SmolLM3-3B**, revision
`a07cc9a04f16550a088caea529712d1d335b0ac1`. Its official weights and identity
receipt are already present on Vista. Previous both-task CUDA/LoRA/checkpoint
checks passed, and four earlier single-state jobs completed in roughly4.8--5.4h.
That is feasibility evidence, not current-runtime validation or new confirmation.

Run **16 fresh initial parents:8 math and8 code**, matching the number of seeds
per stream in the existing32-run confirmation. Freeze new collision-checked seeds,
nonthinking chat configuration, the rendering date and all manifests. Retain K=3,
size2,50 steps, eight correction attempts, three native continuations per candidate
and independent128/128/256 scoring. The block adds48 candidates,144 continuation
draws and208 scored checkpoints. The final fresh collection then contains
48 parents across three models, with eight per each model/task stream.

Revalidate masking, restore/rollback, determinism, code sandbox and both task
families against the current source. Fit **no forecaster** and transfer no
model-ID admission rule to an unseen family. Report SmolLM's selected/uniform
G1/G2, role-level changes and paired continuation loss. Treat the continuation
loss as the predeclared extension endpoint; do not redefine the original
Qwen--Llama primary or select the model revision after observing outcomes.

This addresses **two-family coverage**, not the3--4B scale boundary. A new8B
model introduces download, compatibility and runtime uncertainty while leaving
the main selection/recipe objection unanswered. Do not add it or T=5 rollouts
to this500-SU plan. The four old SmolLM states remain historical evidence.

Larger-model alternative under consideration: the
[7B/8B cost assessment](docs/larger_third_model_cost_20261024.md) prices an
8-run Mistral-7B extension within500 additional SUs, with200 for that extension
and62 for contingency. It trades replication precision for larger-model
coverage. This alternative is not yet selected or submitted; the proposed
manifest still describes16 SmolLM runs.

## E. Implementation and immediate launch after approval

Implement in the current Third Eye repository; preserve immutable old releases.

- `src/third_eye/experiments/revision.py`: validated inputs, factorial and recipe
  arms, run/shared-item inference, role/transition/format diagnostics.
- `experiments/revise_paper.py`: prepare, analyze-saved, preflight, controls,
  recipes, third-model and final-report commands; checkpoints immutable.
- `experiments/jobs/revision_*.slurm`: CPU analysis/validation, integrated GPU
  checks, the three independent research arrays, and final CPU reporting.
- Submission helper: reserve every array element and retry before sbatch;
  persist job IDs, dependencies, source hashes and all failure records.
- Targeted tests: receiver restore and identity, exact pairings and seeds,
  step budgets, gold problem identities, pooling uniqueness, tie handling,
  terminal equality, shared-item resampling, and hard additional500-SU accounting.

On approval, submit the complete chain in one session. CPU validation/manifest
preparation feeds the integrated compatibility/cost check. After that, controls,
recipe tasks and the third-model collection become independently eligible;
none waits for a favorable scientific result in another block. Saved-data
diagnostics run in parallel with GPU queueing. Final analysis depends on all
required outputs. No later routine launch requires repeated approval or a live
SSH controller. Structural failures preserve evidence and isolate the defect.

Use at most20 concurrent allocated GPUs globally, reallocating spare slots as
streams finish. Initially cap controls at6, recipes at6 and third-model tasks
at8. Submit small scheduler-only CPU handoff jobs after array completions to
release spare concurrency while keeping the sum at20; no live local controller
or GPU allocation spent waiting for a slot is needed. All scientific computation, tokenization, model downloads and
hash scans of large artifacts run inside Slurm allocations. Login work remains
scheduler metadata, staging and small receipts. Prefer scratch for new artifacts:
the live work filesystem is about92% full. Do not delete existing evidence.

Target approximately2--3 days for implementation, allocated execution and final
revision, excluding unpredictable queue delays. This is an estimate, not a
deadline guarantee. No architecture search, forecasting recovery or recurring
scientific release decisions are part of the plan.

## F. Paper deliverables

Put uniform/selected G1/G2, per-role/per-stream results, observed sign transitions,
factorial controls and recipe effects in the main text. Keep complete adjusted
tests and runnable artifact instructions accessible. Verify Table1's header-to-
column mapping against the generated PDF and manifests before changing it; the
current source maps Math to GSM8K and Code to MBPP. Explain the date-template
confound in two sentences and remove internal planning narrative and repetitive
caveats. Preserve essential scope and inference qualifications in Limitations.

Treat controls, recipe contrasts and third-model results as separately declared
revision evidence, with their own adjusted comparison families. No confirmation
effect is relabeled or retrospectively replaced. If selection or the recipe
explains the original contrast, revise the claim accordingly. Produce one
integrated results package and revised ARR manuscript after the scheduled work.
