# Scoring provenance and timing: completed review

Checked October 22, 2026 at approximately 13:04 IST. All submitted jobs completed
with exit 0. No final-suite predictions or accuracy evidence was generated.
This closes the declared provenance/cost check, not admission confirmation.

## Execution

| Job | Work | Allocated elapsed seconds |
| --- | --- | ---: |
| 1086306 | Full CPU validation | 75 |
| 1086307 | Provenance and saved-checkpoint audit | 622 |
| 1086308_0 | Llama code inference benchmark | 107 |
| 1086308_1 | Llama math inference benchmark | 178 |
| 1086308_2 | Qwen code inference benchmark | 115 |
| 1086308_3 | Qwen math inference benchmark | 231 |
| 1086309 | CPU timing review | 9 |
| 1087198 | Fixture-classifier full CPU validation | 76 |
| 1087199 | Checked synthetic-fixture classification | 12 |

The four GPU jobs started together at **11:53:50 IST** and the last finished at
**11:57:41 IST**. Review completed at **12:00:34 IST**. Queue delay, rather than
the allocated benchmark work, dominated elapsed waiting. The classifier suite
passed **194 tests in 61.68 seconds**, plus lint.

All artifact scanning, verification, inference and review ran in Slurm
allocations. Login access was limited to staging, scheduler/accounting metadata
and report retrieval. No jobs remain queued at the live check.

## Exposure interpretation

The raw audit scanned **46,591 artifacts** with no unreadable files and verified
all eight saved benchmark checkpoints. Its flags arose from synthetic test
fixtures, including real-format dataset IDs and mock final-score manifests.

The separate classification bound the original report, completion and inventory
to hashes; required each file's original hash, a successful owning validation
receipt, and known mock content. It classified **180 synthetic artifacts** and
left **zero flagged research artifacts**. Paths alone were insufficient. The
raw report remains unchanged and is interpreted alongside this addendum.

The supported statement is **no recorded research exposure in the searched
surviving artifacts**. This does not prove absence of deleted, external,
unlogged or pretraining exposure. There is no automatic declaration that a
future confirmation suite is sealed, and no item pool was changed by this check.

## Measured inference cost

Each stream used the frozen seed-12142 parent and the first candidate's original
continuation, with 16 development items per role on each checkpoint. The jobs
used saved configurations, generation batch size 32 and maximum 512 new tokens.
They performed zero optimizer steps. Completions/correctness were not retained
as new scientific outcomes. Loading, restoration, warmup, generation and
verification costs were recorded separately.

The allocated jobs finished in approximately two to four minutes each. The
already computed review scales the slower per-item rate of the two checkpoints
to the existing final-suite counts, adding 25%:

| Stream | Conditional seconds per single full-suite checkpoint |
| --- | ---: |
| Llama code | 1,059 |
| Llama math | 4,831 |
| Qwen code | 843 |
| Qwen math | 6,237 |

These are **conditional extrapolations**, not measured final-suite runtimes or
complete parent/confirmation prices. They exclude repeated model loads and
must be multiplied by every actually required parent/candidate/continuation
checkpoint. Larger-item completion lengths and verifier costs can differ.
In particular, code final OOD is HumanEval rather than the MBPP development
proxy. Math scoring at full final-suite size is materially more costly than code;
do not assume additional inference is free or that a full-suite expansion fits.

## Budget and evidence decision

Final accounting, including minimum allocation billing, charges **1.2475 SUs**
for the original scoring campaign and **0.165 SU** for classifier validation and
classification: **1.4125 SUs combined**. B development totals **145.6106 SUs**
against 192. The bounded-additions project ledger is **233.4067 SUs**, leaving
**966.5933 SUs** under 1,200. This is project-cap headroom, not an account balance
or a retroactive budget for all historical campaigns.

The review's raw status still reflects the original unresolved fixture flags;
the checked classification resolves those flags within the searched scope.
Its other stop conditions remain: no sufficiently powered and fully priced
confirmation design has been frozen. No C, recursion or scaling was submitted.

Next design review must jointly choose independent scoring items and fixed
balanced N, count all inference/training/validation costs and verify practical
superiority plus retention precision. Price optional recursion last. The
exploratory backbone-H2 baseline remains a candidate baseline, not a promoted
policy or established forecasting success. These checks add execution and
provenance confidence, not new evidence of admission effectiveness.

Local artifacts: `runs/deployment/evidence/scoring_precision_20261022/`.
Verified review report SHA256:
`63f861a2f6f6395999b45c8960fd5218d3acf7561f691ca85c8535042fc02e05`.
Verified classification report SHA256:
`1003b692b68d64e8534694eecc88adb2687e8a9becdcff8b3729101077d60a71`.
All four benchmark reports were verified against their completion receipts.
