# Fresh admission development pilot

Declared October 18 before collecting any new parent outcomes. This executes
stage B of [research.md](../research.md), following the completed Stage A review.
All earlier source releases and evidence remain unchanged.

## Fixed collection and information boundary

Collect 16 new initial states: Llama-3.2-3B-Instruct and Qwen3-4B, each on math
and code, seeds 12142, 12143, 12144, 12145 in each stream. An allocated CPU job
audits historical run metadata for seed collisions before writing configurations.
Use one offered generation per parent, K=3 distinct size-two verified candidate
batches and the existing frozen A2 update/harvest budgets. Overlap across
candidate batches is allowed; duplicated problems inside a batch are forbidden.

Collect **three total** continuation draws for each candidate: the original
continuation and offsets 50,000,000 and 100,000,000. All draws start from the
same saved candidate M1; only M1 is eligible for trajectory commitment. Preserve
correction-scarved terminal M2=M1 branches with zero continuation optimizer steps.
No terminal-rate cutoff, replacement seeds, automatic retries or dynamic budgets.

Freeze `date_string="11 Oct 2026"` from the beginning of harvesting through all
probes, training and evaluations. Qwen additionally retains `enable_thinking=false`.
Store the rendering context with correction pools and precommit records and
validate it against item artifacts and checkpoints. Historical Llama data are
not promoted to clean fresh confirmation.

Use the existing hashed development suites: target 64, OOD 64, retention 256;
harvest training pools remain math 1,024 and code 306. These item sets are
development data. Audit final-suite identities/counts/hashes on CPU, without
reading or generating final predictions. File disjointness does not prove those
items were never historically evaluated; resolve that exposure before C.

Stage A did not establish stable recipient/donor interactions. Consequently B
has **no pooling arm** and no new candidate-construction intervention.

## Frozen developmental analysis

Reuse the historical deterministic hash halves. Pick the immediate candidate
using one half; score the other. Uniformly average exact candidate ties and
both orientations inside each parent. For the privileged admission proxy,
choose admit/hold on two continuation draws and score the third on the other
item half; average all draw rotations within parents. Report signed outcomes,
not the positive part of a noisy scored gain. The proxy is not an oracle or an
upper bound.

Compare forced immediate selection, hold, protected replay, H1-trained and
H2-trained ridge admission rules. Use the same four features for both rules:
selection-half target/OOD/retention H1 gains and log1p of the actual scaled-BA
candidate-minus-parent norm. Norms use compact Gram products of CPU-loaded
LoRA weights, without base-model weights, SVD or short-probe factor proxies.
Archive existing probes but exclude them: their retention anchor includes
scoring items. Model identity, seeds, paths and future outcomes are not features.

For each item orientation, leave one whole parent out. Use parent index +4 mod16
as validation and the other 14 as training. Alphas [0.1,1,10,100], thresholds
[0,.0025,.005,.01], predicted retention floor -.005; alpha minimizes validation
head MSE (tie: larger alpha), threshold maximizes validation utility for its
training horizon (tie: larger threshold). Training and validation targets use
selection-half items only. The held-out scoring half never fits either gate.

Development horizon choice: prefer H1 unless H2 exceeds its parent-averaged
cross-validated scoring utility by more than 0.5 percentage points. Report both,
including replay contrasts, coverage, observed beneficial rejections and harmful
admissions. This choice uses B outcomes and must not be called confirmation.
It is provisional until gate coefficients and an exact C design are frozen.

Use 5,000 parent bootstrap samples, four parents sampled within each fixed stream.
Also report conditional shared-item uncertainty with identical role/item pools
resampled together. These are conditional intervals, not a simultaneous joint
confirmation test. Show power sensitivities under both overall-parent covariance
and within-fixed-stream residual covariance, with the same variance multipliers;
neither includes fitted-policy or entirely new-item uncertainty. No favorable
projection alone authorizes C.

Stop after the complete 16-parent report for review of headroom, utility against
both constant policies/replay, stream heterogeneity, power, item exposure and cost.
Headroom below the practical 0.5 pp point target stops C; unresolved uncertainty
does not prove ideal admission impossible. No automatic C, recursive rollout,
additional backbone, architecture sweep or restored forecasting claim.

## Execution and budget

Transfer 72 SUs from the existing contingency envelope to B **before B outcomes**:
B ceiling becomes 192, contingency 78; C560 and D250 remain ceilings. Total
new planned ceilings remain 1,088 and the persistent ledger hard cap stays1,200.
Previously charged work remains charged; account availability does not raise it.

CPU preparation subtracts only recorded harvest/training time for the two omitted
draws from each complete historical five-draw pilot. It retains their other
overhead, takes the worse of the two benchmarks in each stream, adds25%, and
rounds each stream limit up to15minutes. Reserve four jobs at each stream limit,
plus validation/preparation/CUDA/controller/review (maximum1.8525SUs overhead).
If total exceeds192, stop before GPU pilots. New-parent scarcity and token lengths
can still exceed measured runtimes; a timeout is preserved evidence, never an
automatic retry.

Dependency chain: full CPU suite/lint -> allocated CPU preparation -> CUDA tiny
adapter/terminal checks and real preflight in all four streams -> controller ->
four GPU arrays of four parents -> allocated CPU structural/statistical review.
At most16 GPUs run concurrently. Reserve before every submission and retain all
ambiguous reservations. Failed validation blocks downstream launches. All
computation runs under Slurm; login access only stages files, hashes, metadata
and scheduler requests. All queued jobs continue independently of local SSH.
