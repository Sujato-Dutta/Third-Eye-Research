# Positive-regime confirmation — frozen November 9, 2026

## Question and development evidence

Does a verified self-update that improves immediate accuracy remain useful after
another update? Test the selected recipe without changing it in response to final
outcomes. This is a prospective extension of the existing study, not a new
forecasting experiment. Earlier results and the existing paper archive remain
unchanged until this cohort is reviewed.

All eight development parents completed. The frozen screen selected Qwen3-4B,
recipe r1: eight distinct verified corrections, 25 AdamW steps, LR 3e-5. Across
its four development parents, equal-role immediate gain was +2.132 pp and
retention change +0.342 pp. These are development estimates. The other Qwen
32-correction recipes were unavailable; no batch size was silently reduced.

## Fixed cohort and arms

Twenty fresh parent experiments, seeds **2026110900–2026110919**, math only,
Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
These are independent harvesting/adapter-initialization/optimization seeds,
starting from the same pretrained backbone; they are not twenty distinct
pretraining checkpoints. No seed replacement or outcome-dependent sample-size
change is permitted. No additional family or scale claims follow from this
single-regime confirmation.

Each parent has one prespecified first update, not a best-of-three selection.
The shared legacy configuration still validates K=3 for the historical runner;
this dedicated runner does not invoke candidate selection or create K=3 labels.
Its actual batch size and single-update design are recorded explicitly.

Score M0 (Hold), M1, and three paired draws of each continuation arm:

| Arm | Starting model | Donor corrections | LR | Steps |
| --- | --- | --- | ---: | ---: |
| Native | M1 | Generated from M1 | 3e-5 | 25 |
| M0-direct | M0 | Identical to paired Native | 3e-5 | 25 |
| LR-decay mitigation | M1 | Identical to paired Native | 1.5e-5 | 25 |

All three arms share correction identities, complete text, order and optimizer
seed. Every update starts a fresh optimizer. The lower-LR arm isolates second-step
update strength; it was **not** selected because of an observed H2 advantage.
The development screen did not compare mitigations. This explicit sensitivity
choice does not claim to be an optimized policy.

Retain rank-16/alpha-32 attention LoRA, BF16, microbatch 1, accumulation 8,
1,024-token complete SFT targets, generation length 512, greedy/sample batches
32/16, temperature 0.7, frozen chat date `11 Oct 2026`, thinking disabled, and
the original 1,024 training problems. Harvest at most eight correction attempts
per failed problem. Only verifier-passing, length-compatible corrections enter
batches. Problems are unique within each batch. No gold solutions enter research
updates. A training-only one-step compatibility fixture is rolled back exactly.

Initial batch selection uses seed+17, first-update optimization seed+101.
Continuation harvest seeds use parent seed plus 10,000/110,000/210,000;
donor selection uses draw seed+19 and all paired optimizers draw seed+31.

If the initial batch cannot supply eight corrections, retain the parent as
M1=M0 and all continuation arms as identity states. If a later draw cannot
supply eight, Native and mitigation preserve M1 and M0-direct preserves M0.
Record availability=0, retain the draw and parent, and report scarcity frequency.
Do not clone corrections, shrink batches, or replace terminal parents.

## Evaluation and inference

Use the already frozen expanded manifests: **1,319 GSM8K test, 500 MATH-500,
1,024 MMLU test** items. Accuracy utility is the equal mean of these three roles.
Development items remain separate. Some final items were evaluated historically;
the expanded pool is not described as wholly unseen. Fresh experiment outcomes
are never used to choose recipes, items or hyperparameters.

Seal every adapter and donor batch before scoring any final item for that parent.
Store item predictions, verifier outcomes, model/template/source hashes, optimizer
budgets, and timing. Identical adapters may reuse deterministic scores, with an
explicit identity-reuse receipt; no distinct adapter borrows another's score.

The four primary endpoints form one simultaneous 95% interval family:

1. G1 = U(M1) − U(M0).
2. Native increment = U(Native M2) − U(M1).
3. Native-minus-direct level = U(Native M2) − U(Direct M2).
4. Increment interaction = [U(Native M2) − U(M1)] − [U(Direct M2) − U(M0)].

The interaction addresses the unequal amount of prior training: a negative
Native-minus-direct level alone does not identify parameter interference.
Matched donors control data differences between these arms, not every effect
of prior updating, nor endogenous donor quality relative to an M0-harvested arm.

Average the three continuation draws **within parent**. Jointly resample parents
and shared evaluation items, preserving item identities across arms and parents.
Use 5,000 bootstrap replicates, seed 20261109, with Bonferroni intervals within
the four-endpoint family. A separate two-endpoint secondary family covers
mitigation utility gain and retention change versus Native. The retention
noninferiority margin is −0.5 pp. Report per-role ordinary 95% descriptive
intervals, native G2, direct gain, scarcity and timing. A sub-1-pp half-width is
a precision goal, not a promised result. Require all twenty parents for the
confirmatory report; failed or partial jobs do not disappear from the ledger.

Positive G1 requires its simultaneous lower bound above zero. Evidence of
delayed loss requires the Native increment's simultaneous upper bound below
zero. Interpret the interaction and direct contrast independently; an ambiguous
result remains ambiguous. No forecasting or admission-success claim is implied.

## Execution and storage

All work runs on DGX compute nodes through Slurm, one allocated A100 at a time.
No new Vista jobs or SUs. Run complete CPU validation first, then two operational
pilots (included in the fixed twenty), then the remaining eighteen. A CPU timing
and storage check between these blocks may stop unsafe execution but cannot
change scientific settings. Final CPU analysis waits for every parent to pass.

The first pair measures runtime on the expanded pools; prior small development
scores are not a reliable completion estimate. Jobs have a protective 48-hour
per-parent limit. The scheduler mask is preserved. The 50-billion-byte whole-home
policy applies throughout. Losslessly compact completed adapters and verify all
hashes. If necessary, retire only checksum-verified pretrained Llama-3.2-3B cache
weights after its development cohort has finished; preserve all results,
adapters, tokenizer/config metadata and environments. Re-downloadable weight
hashes and source provenance remain recorded. No automatic further expansion.
