# Protocol Amendment A1: Correction Scarcity

Status: frozen before new A1 experiments, October 3, 2026.
Scope: verified correction harvesting, candidate composition, and standardized
H=2 continuation availability. Original runs and their protocol remain archived
unchanged. Correction scarcity is an operational edge case, not a failed Gate 1
or a validated saturation claim.

## Fixed rules

K=3 candidate batches may overlap across candidates. Every batch contains two
unique training examples, and the three example compositions must be distinct.
A1 samples compositions without replacement from the space of size-two subsets;
examples may occur in several subsets. Difficulty strata are sampled in
proportion to their counts in the verified pool, and stratum quotas may differ
between candidates. Three identical difficulty histograms are not required.

Candidate size two, learning rate 0.0001, LoRA rank 16/alpha 32, dropout zero,
50 optimizer steps, micro-batch one, accumulation eight, sequence limit 1024,
and gradient clipping one remain unchanged. There is no dynamic batch shrink,
correction cloning, unverified completion, or extra training step. All available
continuations use the same full update budget and standardized procedure/seed.

The enlarged harvest scans the predeclared seed-42 portion of GSM8K-train:
1,024 examples following the existing 64-example development reservation, before
the existing overlap audit. MBPP scans its entire official train split, before
the same audit. Development and final benchmark files remain byte-identical to
the original suites; only the training manifest changes. Original dataset
revisions are reused. Excluded duplicates are recorded, with no replenishment
from validation/test data and no outcome-based prompt selection.

Every initially failed training problem receives at most 16 correction attempts,
stopping at its first verifier-passing correction. Generation remains limited
to 512 new tokens at correction temperature 0.7. The full fixed training portion
is scanned; only verified failure-to-correction examples survive. Harvest costs,
remaining failures, attempted revisions, and verified pool sizes are recorded.

After a candidate's t+1 update, harvest again using the same procedure. If the
pool cannot supply two unique verified corrections, declare the continuation
correction-starved/terminal and set M_(t+2) = M_(t+1). Record
`continuation_available=0`, `terminal_reason=correction_scarcity`, zero executed
continuation optimizer steps, and the unchanged adapter hash. H=2 consequences
then equal H=1 consequences exactly. Otherwise `continuation_available=1` and
the full standardized continuation is trained. Candidate updates always retain
the full 50-step budget.

Continuation availability and realized continuation statistics are outcome
metadata. They never enter pre-commit forecaster inputs. Report their frequency
and terminal/nonterminal descriptive slices without excluding terminal branches
from the primary matched-state analysis.

If a parent pool cannot form three distinct size-two compositions, preserve
the state as `correction_scarcity` and end that trajectory. Do not publish an
incomplete K=3 state, invent missing labels, or classify scarcity as a negative
phenomenon gate.

## Pilots and Gate 1

Start four fresh one-state H=2 pilots at seed 1042: Qwen math/code and Llama
math/code. No old accepted checkpoint is resumed or overwritten. This seed
matches the original diagnostic trajectories rather than selecting a previously
successful seed.

Gate 1 uses the existing minimum of 20 complete matched states. After these four
pilots, the predeclared additional calibration seeds are 2042, 3042, 4042, and
5042 for each stream. These are fresh one-state runs, capped at four concurrent
GPUs, to assess the phenomenon without requiring every trajectory to reach T=5.
Do not replace missing states with outcome-selected reruns. If fewer than 20
complete states exist, report `insufficient_evidence`, not a failed phenomenon.

Software/source checks and deterministic CUDA verification must pass first.
Primary Gate 1 thresholds remain ranking disagreement >=20% OR harmful
component updates >=15%. Confidence intervals and correction-scarcity counts
are retained. Gate 2 thresholds remain held-out H=2 Spearman >=0.30 and
informative top-1 >=45%, above tie-adjusted chance, with the existing evidence
minimum. No gate threshold is weakened to obtain a passing result.

The A1 controller stops for review after Gate 1, even if it passes. The 20-GPU
study, forecaster training, transfer, and scale expansion remain held. A1 source,
manifests, amendment hashes, model proofs, scheduler records, and outputs are
versioned separately; original and A1 labels cannot be pooled.

Basis: `ACL27_Third_Eye_Research.docx`, sections on verified self-corrections,
equal-size stratified candidate batches, standardized H=2 continuation, research
gates, and the anticipated risk of too few verified corrections.
