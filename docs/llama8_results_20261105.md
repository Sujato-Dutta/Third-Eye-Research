# Llama3.1-8B complete results

November5,2026. All eight parents (four code/four math),104 scored checkpoints,
72 continuation draws and analysis1114003 completed successfully. All eight
completion/retention receipts passed source/plan checks; zero draws were terminal.
The downloaded report hash matches its passed analysis receipt.

Source SHA256: `8282f1d02158f3489c2791d1c2f492040e12eff44c197cf424b61e7a499a0ef8`.
Plan SHA256: `47a7e9ff91e464c1d45406a798e1a0a5122f690c700470df927ae473b0ec66a4`.
Report SHA256: `9e2695d9813ebf0ad52108de7594f18632c83dd7b0e36d87c5e6775c84db2713`.
Local evidence: `runs/deployment/evidence/vista_llama8_20261105_final/`.

## Aggregate endpoints

Utility is the equal-weight mean of target, OOD and retention accuracy.
All values are percentage points. H1/H2 gains are relative to the parent;
change is H2 minus H1. Selection uses development outcomes only, with uniform
tie weights. Joint bootstrap intervals resample parents within task, shared
exact scoring-item identities and nested continuation draws. The two
model-specific scale primaries use97.5% intervals for joint95% coverage.
Other endpoints are exploratory.

| Endpoint | Mean |97.5% interval |
| --- | ---: | --- |
| Selected H1 gain | -3.418 | [-6.364, -0.716] |
| Selected H2 gain | -5.919 | [-8.678, -3.315] |
| **Selected H2 minus H1 (primary)** | **-2.501** | **[-4.411, -0.583]** |
| Uniform H1 gain | -4.199 | [-6.467, -2.051] |
| Uniform H2 gain | -7.559 | [-10.047, -5.092] |
| Uniform H2 minus H1 | -3.360 | [-4.716, -1.869] |
| Replay H2 gain | 0.000 | [0.000, 0.000] |

The replay endpoint equals hold in this report. It does not improve over
holding; inspect the frozen commitments before attributing this to a specific
pattern of admission decisions.

## Role decomposition

| Role | Selected H2 minus H1 |97.5% interval |
| --- | ---: | --- |
| Target | -4.297 | [-8.545, -0.130] |
| OOD | -2.376 | [-5.697, +0.863] |
| Retention | -0.830 | [-2.441, +0.708] |

These head-level summaries are exploratory. Their means do not establish a
mechanism; the OOD/retention intervals cross zero.

## Interpretation

The declared continuation-change primary is negative with an interval excluding
zero. Immediate utility is already negative: the result establishes further
deterioration under continuation in this configuration, not immediate
harmlessness. Uniform-candidate summaries also show deterioration descriptively.

The five-model study supports evaluating immediate and downstream consequences
together. The Qwen8B primary remains inconclusive; do not claim universal harm
or a causal effect of model size. This eight-parent cohort has four seeds per
task and remains scoped to size-two,50-step LoRA, K=3 and H=2. It does not
establish long-horizon RSI behavior, a successful mitigation or a causal mechanism.

## Completion, runtime and cost

Code runs took7h11m--9h05m; math runs took9h44m--12h06m. Final analysis took24s.
The replacement execution chain consumed77.743056 SUs, including calibration
and CPU checks. The complete additional Vista revision consumed296.372500 SUs,
including earlier failed checks. This excludes historical project costs and
DGX GPU time. Unused reservations of248.696944 SUs were released. The account
independently reported1142 available SUs at final inspection.

All64 planned fresh parent runs across five models and their model-level
analyses are complete. The sixteen matched control follow-ups are also complete;
they reuse inspected parents and do not increase the fresh-parent total.
Remaining work is integrated saved-data analysis, manuscript/figure/table
revision and artifact verification. No additional GPU experiment is scheduled
or automatically authorized by this completion.
