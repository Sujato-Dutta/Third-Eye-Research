# Provisional remaining-compute projection: October12

This is a planning estimate, not a frozen sample-size decision, spending
authorization or a guarantee of a positive scientific result. The current
eight developmental pilots remain unchanged and larger collection remains
held for inspection. The target submission is late December.

## Observed execution at18:08 IST

Two of eight pilots completed: both Llama/code seeds. Their allocated runtimes
were30,734s (8.5372h) and35,036s (9.7322h). Six others were RUNNING with roughly
9.7–10.0h elapsed. Fifteen of24 candidate records were complete; no failure.json
artifact was present. Math pilots are incomplete and must not be assigned the
code pilots' mean completion time. All scientific work is in Slurm allocations.

`taccinfo` reported2,195 available SUs inIRI23021 at18:08 IST. Account balances
can lag running/recent job charges. These observations are stored in transport
responses672–674; scientific inspection1065436 remains dependent on the array.

The [official Vista guide](https://docs.tacc.utexas.edu/hpc/vista/running/)
lists1 SU per GPU-node hour forgh/gh-dev and0.33 per CPU-node hour forgg, with
a15-minute minimum per job. Queue wait time is not allocated running time.
Count each array task once; do not additionally bill its parent array record.

## Illustrative two-model completion envelope

| Remaining work | Assumption, not yet a frozen design | Estimated SUs |
|---|---|---:|
| Finish current pilots | Incomplete math/code branches; complete records retained | 50–120 |
| Larger balanced label collection | Illustrative100 additional states, whole-trajectory partitions; assume12–20h/state forK=3 and five draws | 1,200–2,000 |
| Independent scoring | Fresh seeds and disjoint items on untouched evaluation trajectories | 200–400 |
| Recursive policy comparisons | Expected-value/H1/greedy/random plus no-update throughT=5; online runs do not generate the fullH2 tree | 200–500 |
| One mechanism intervention and final evaluation | Bounded matched experiment, final metrics | 100–250 |
| Forecaster fitting, ablations, statistical review, archival and validation | Small forecasters/analysis mostlyCPU; these costs need separate accounting | 10–30 |
| Total before contingency | Sum of the assumptions above | 1,760–3,300 |

For planning, round this to approximately **2,000–3,500 SUs from now**. The
central working allowance is about3,000, not a committed experimental count.
The100-state example is not claimed to be statistically sufficient: the pilot
variance, desired minimum useful selection improvement and trajectory-level
precision/power review determine the final sample size. A need for more states,
slower later-generation harvesting, repeated implementation failures or expanded
confirmation would increase cost beyond this envelope.

A proper third-backbone confirmation or8B study is excluded from the core
envelope and requires separate profiling/reservation. The prior four-state
SmolLM sanity check is not an adequately powered replication. Writing itself
does not require a substantial GPU allocation.

At the reported2,195 balance, a3,000-SU allowance has a nominal805-SU gap,
before outstanding billing and contingency. A planning reserve of roughly
**1,000–1,500 additional SUs** would make that envelope safer; it is not yet an
established requirement. A narrower study may fit the current allocation, but
reducing required statistical precision merely to fit the budget is not an
acceptable basis for a superiority claim.

After all eight pilots finish, replace the assumed12–20h/state range with
measured stream-specific accounting and separate harvest/evaluation components.
Run the power/precision calculations on an allocatedCPU node before freezing
the balanced cohort. Do not release a larger array from this estimate alone.

## Measured update: October13,10:08 IST

All eight one-state pilots and the structural inspection completed successfully.
Total pilot allocation time is108.6611 GPU-node hours, approximately109.08 SUs
including the validation/inspection minimum charges. The balanced mean is
13.58h per state; individual states range8.54–19.72h. The illustrative100-state
collection would cost approximately1,358 SUs if it retained this task balance
and initial-state runtime distribution. This is an extrapolation, not evidence
that later generations have the same cost or that100 states provide enough
statistical power. The prior2,000–3,500 remaining-study envelope remains a
planning range until whole-trajectory sample sizes and independent scoring are
frozen. The live allocation balance is now2,093 SUs. No larger campaign has
been released automatically.
