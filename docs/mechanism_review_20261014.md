# Completed mechanism review and bounded next analysis

At22:25IST on October14, all eight1068803 GPU jobs and CPU review1068804 were
complete with exit0. There are24 complete intervention candidates and72
continuation draws, with0terminal continuations and0failed states. Every
parent retains K=3, matched verified candidate data/seeds,50steps, baseline
continuation LR and the declared A2 harvesting/scarcity rules. The experiment
changed only candidate LR from1e-4 to5e-5. These are eight previously inspected
development parents, not untouched confirmation or final Gate2 evidence.

Utility is the equal-head mean of target/OOD/retention accuracy deltas. Values
below are utility percentage points, not target accuracy alone.

| Declared comparison | Mean pp | Descriptive95% interval pp | Holm p |
|---|---:|---|---:|
|Primary: half-LR minus baseline, holding baseline-immediate candidate choice fixed|−0.24|[−0.97,+0.37]|1.0|
|Secondary: half-LR minus baseline, average over every candidate/draw|+0.98|[+0.32,+1.71]|0.3125|
|Secondary: half-LR immediate reselection minus baseline immediate selection|−2.20|[−5.35,+0.12]|1.0|
|Guarded rule minus forced immediate selection|+0.68|[−0.70,+2.33]|1.0|
|Guarded rule minus holding unchanged|−0.29|[−2.56,+1.45]|1.0|

The primary mechanism benefit is not established. The positive all-candidate
average is secondary, nominally positive and unconfirmed after the declared
multiple-comparison adjustment; it must not replace the primary endpoint.
Reselection and the guarded rule also show no established practical gain.
Thus this stage does not restore the forecasting story, pass Gate2 or justify
automatically releasing large recursive/GPU experiments. It does not establish
that smaller updates never help or that future value cannot be predicted.

## Provenance and accounting

Immutable execution sourceSHA:
`ce71b7956a1806a086838bbb89d4554d0043428f863513c0c1239a0fb2e9c624`.
Native first-wave reportSHA:
`a083df786b4012271855bf7093aaa2f8de9fc18a664c5d987f35dd7629e1e1c2`.
Campaign: `/scratch/11617/sujato_ts/third_eye_research/runs/bounded_additions_ce71b7956a18`.
Receipts and report copies: `runs/deployment/evidence/bounded_additions_20261014`.

The GPU array used60.9894node-hours, with per-parent runtimes5.05–11.92hours.
Including checks, CPU analysis and superseded CPU checks, the additional
consumption is61.65194SUs. Cancelled unstarted GPU jobs cost0. Remaining headroom
under the user’s additional1,200-SU cap is1,138.34806SUs. The account display
reports2,032SUs available; this includes resources outside the new envelope.
The ledger now uses completed accounting rather than the earlier reservations.

## Next analysis: explicitly exploratory, CPU only

Before exposing its output, declare this small diagnostic. It consumes the
completed report and checkpoint-bound per-item outcomes only. It changes no
models, labels, training, seed grids, candidate sizes, terminal rules or primary
endpoints. The first-wave report remains immutable.

1. Decompose the matched candidate-LR effect by target/OOD/retention, each
   four-stream group and each candidate. Report uniform candidate averages
   separately from the effect weighted by baseline immediate selection.
2. Report all three existing continuation-draw effects within each parent,
   averaging disjoint item-fold orientations as repeated measurements. They
   remain eight independent development parents. Subgroups of two parents are
   descriptive and cannot become new confirmatory discoveries.
3. Check measured adapter parameter-change norms. Changing LR does not imply
   a pure intervention on effective weight magnitude or a precise50% norm change.
4. Give normal-approximation sample-size/precision projections for0.5,1and2pp
   hypothetical effects, balanced in multiples of four parents, using the
   completed parent-level standard deviations. Compare16/24/32/48/64parent
   precision. These are planning sensitivities with uncertain pilot variances,
   not proof of power or sample-size claims for T=5 online experiments.
5. Retain the384-SU future-collection allowance as unsubmitted. A24h task slot
   is not automatically a new independent trajectory, and a fresh paired LR
   replication may require both control and intervention collection. Actual
   science, seed split and cost must be frozen before a new GPU submission.

Run full scheduled CPU validation followed by one30-minute CPU diagnostic.
Maximum new reservation0.275SUs, counted in the same persistent1,200-SU ledger.
No automatic GPU follow-up, endpoint switch, new architecture or claim of a
beneficial deployed policy follows from this exploratory diagnostic.

## Submitted diagnostic

Full scheduled CPU validation1071698 and diagnostic1071699 were accepted,
with the diagnostic held byafterok:1071698. Both use thegg CPU partition;
no new GPU task was submitted. SourceSHA:
`1c5d0e113752cc42ed49d5ce0db590fe5b9fcd47615a9d9db6405422bb6ce35e`.
Output: `/scratch/11617/sujato_ts/third_eye_research/runs/mechanism_precision_1c5d0e113752`.
Maximum additional reservation0.275SUs. Two synthetic panel checks and lint
passed locally. The source-bound first-wave primary remains unchanged.
