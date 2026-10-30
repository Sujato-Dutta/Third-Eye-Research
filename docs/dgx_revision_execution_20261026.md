# DGX review-control execution

**October29 evening:** all16 controls and analysis28974 completed successfully;
all272 cells and544 scored-file hashes passed. [Results and the independent
format/exposure audit](dgx_controls_results_20261029.md) document inconclusive
recipient/recipe contrasts and a negative, recipe-conditional gold-source
effect. Storage rotation29449 and Qwen8B preparation/preflight29450/29451
passed. Qwen array29452 has started; Llama29456 follows its completed block.
The read-only audit needed a local tokenizer metadata-path correction;
replacement29581 passed, preserving the failed nine-second29578 attempt.

**October29,10:29 IST live check:** fifteen of sixteen control parents completed,
with255 cells and matching completion receipts. Element15 is running; no
control parent has failed. Analysis28974 waits for this final element. The
two8B chains remain pending their declared dependencies. SmolLM16 and its final
analysis on Vista are complete; selected H2-H1=-4.262pp,95% interval
[-6.163,-2.581]. Its H1 is already negative; do not infer neutral immediate
effects across all three model families.

**October28,20:03 IST live check:** thirteen of sixteen control parents completed,
with221 scored cells in source/registry-matched completion receipts. Element13
is running, elements14-15 await the one-GPU array limit, and analysis28974
remains dependent. No control element has failed. Vista completed15/16 SmolLM
parents; the remaining math parent finished label collection and is in final
scoring. Both final analyses remain automatic and have not started.

**October28,11:17 IST live check:** eleven of sixteen control parents completed;
their source/registry-matched receipts contain187 scored cells. Element11 is
running on dgxb and elements12-15 await the one-GPU array limit. The completed
set includes all eight Llama parents and three Qwen/code parents. No control
element has failed. All sixteen Vista SmolLM parents are now running, each for
about6.2h, across sixteen GPU nodes. Their bounded log tails show completed
development evaluation and ongoing correction harvesting, with no traceback.
Both final analyses retain scheduler dependencies; no complete-cohort scientific
claim is inferred from this partial execution status.

**October27,20:09 IST live check:** seven of sixteen control parents completed,
with119 scored cells and matching source/registry completion receipts. Element7
is running; elements8-15 await the one-GPU array limit. The completed set is
four Llama/code and three Llama/math parents; Qwen is not yet represented in
the completed subset. No control element has failed. Vista's sixteen SmolLM
jobs remain pending priority, with no estimated start. Scheduler metadata shows
no idle gh nodes (572 allocated, two reserved and two planned). Preserve their
queue age and current reservations; do not cancel/re-submit unchanged jobs.
Final DGX inference remains dependent on the complete cohort.

**October27,09:29 IST live check:** compatibility28972 passed all four streams.
Array28973 completed its first four parents successfully, each with all17 scored
cells; their completion receipts were downloaded and checked against the source
and registry hashes. These four are the Llama/code subset, not a balanced
cross-stream result. Element4 is running on dgxa, elements5-15 await the one-GPU
array limit, and final analysis28974 retains its dependency. No control element
has failed. Vista's sixteen SmolLM parents remain pending scheduler priority.
The68 scored cells are execution progress; no complete-cohort contrast is
inferred before the remaining parents and declared analysis finish.

October26,2026. Key-only SSH access to `dgx-login01` was restored using the
specified local Ed25519 key. The cluster guide is
[DGX_A100_training_Doc.pdf](DGX_A100_training_Doc.pdf), especially pages16 and39.
The student association permits32 CPUs,256GB host memory and one GPU at once;
the maximum job walltime is seven days. The guide lists70GB storage, but the
user's stricter50GB policy governs this deployment. `quota` reported no enforced
quota on the compute node; this does not override the50GB operating limit.

Login-node activity is limited to SSH/SFTP staging, directory metadata and
scheduler operations. Directory scans, deletion, package installation, model
downloads/hashing, tests, analysis, tokenization and training run in Slurm
allocations. Submission occurs from the home directory. GPU jobs request one
full A100,16 CPUs and64GB RAM; the array permits only one concurrent element.

**October26 recovery update:** the transfer completed and import28797
passed, including both pinned model weights. Preflight28798 failed after33s
because `/usr/bin/bwrap` was absent; dependent array28800 and summary28801
were cancelled before allocation. No parent-control cell ran. CPU job28970
installed a user-local Bubblewrap from the configured package repository,
verified the package checksum and passed isolated positive, negative and
early-exit code tests. No host-execution fallback was enabled. Its executable
hash is `bf6cf3d4456665f5d80c14eb79a76e0ce87b39a0883e614a161658b4ec33492c`.
The unchanged source, existing full validation and original frozen inputs now
feed compatibility28972, sixteen-parent array28973 (`0-15%1`) and analysis28974.
These use a new `review_controls_9d722d4af1b3_sandboxfix` output directory;
the original failure and all unstarted cancellation records are preserved.
This is an environment repair; no scientific setting or parent was changed.

## Verified cleanup

Home initially occupied41,985,511,424 bytes. Allocated job28779 removed only a
recoverable pretrained-model download cache and an installer archive, recording
an inventory and preserving small model metadata. It reclaimed20,739,891,200
bytes. Subsequently, the user explicitly requested complete deletion of
`MINDSET`, `MINDSET_v2`, and
`Hybrid-Image-Denoising-for-Gamma-Noise-Removal-cvip26`, including their results
and environments. Job28781 removed exactly those three resolved home-directory
targets and verified all were absent; it reclaimed20,439,183,360 additional
bytes. Other projects, account files and the new Third Eye directory remain.

The allocated inventory job28777 completed. An initial audit wrapper28776
failed immediately because Slurm's wrapper used `/bin/sh`; the successful audit
used an explicit Bash script. A Windows-text submission was rejected before
allocation; subsequent script staging uses byte-safe transfer. These records
are retained. No scientific experiment was run during cleanup.

The isolated environment in `~/third_eye_dgx_20261026/.venv` was installed by
CPU job28780, with Torch2.8.0+cu126, Transformers4.56.2, PEFT0.17.1,
Accelerate1.10.1 and the pinned evaluation dependencies. Home occupied
7,394,197,504 bytes after installation. Job28791 downloaded and verified the
pinned official Qwen3-4B revision,8,060,896,856 bytes including metadata.
Old project environments are not dependencies of this deployment.

## Frozen controls and hardware bridge

The active source release is `9d722d4af1b3`, full hash
`9d722d4af1b38b771e341821221a5d58d96a55d4af6439c37b456704c817ac32`.
Its complete CPU suite passed223 tests and lint on DGX in job28796.
Six focused added tests cover portable checkpoint identity, data deduplication,
full reference solutions, bounded-memory Python3.10 hashing, paired contrasts,
and shared-item uncertainty.

Vista CPU export1101572 verified the original32-parent cohort and the frozen
16-parent subset, selected candidates using only the original development
commitments, and fixed tie/draw choices by SHA256. It exported identical scoring
item outcomes and the exact M0/M1 adapters and verified continuation batches.
The registry hash is
`3af0f503bf0cf507d3d6633d2a4eb3f642b9913553fdacae00c80c37cf013524`.
The818,572,008-byte payload archive has hash
`a50d822016c319163512cea1eb578767e918ed239d458cc9ae63ed5e93b848e9`.
The export reserved0.165 Vista CPU SUs, taking revision charged/reserved usage
to426.0175 SUs, still below500. DGX does not consume Vista GPU SUs.

The user's subsequent allocation instruction is Vista first, up to the existing
500-additional-SU ceiling, with DGX for overflow. The live Vista ledger cap is
now1,042.573611 SUs including the542.573611-SU pre-revision baseline. All
SmolLM jobs stay on Vista; its73.9825-SU remainder remains available there
for finishing work and complete measured follow-ups. The larger paired-control
block is DGX overflow; no task has been launched on both sites. This does not
authorize a full migration: the818.6MB compressed transfer contains only frozen
manifests, selected M0/M1 adapters and saved scoring outcomes. Pretrained model
weights and the Vista project/cache are excluded. Transfer progress does not
mean any DGX control has started; import remains held until checksum verification.

**Execution supplement R-DGX1:** preserve the same16 parents, two fixed draws
per parent, original training/evaluation settings and all proposed controls.
Because the follow-up uses A100 hardware and a different host/sandbox runtime,
recompute every native50-step continuation baseline on DGX and re-evaluate both
receivers there. Do not use the archived Vista outcome as the direct paired
baseline for a new DGX arm. This adds32 native-baseline updates to the original
208 missing/control/recipe updates:240 updates plus32 receiver-only scores,
or17 scored cells per parent and272 in total. Terminal batches remain identity
updates with zero optimizer steps; no parent is replaced or excluded for scarcity.

All four receiver/donor cells use a matched optimizer seed within draw. The
10/25-step and full-reference arms use those same fixed batches and seeds.
Pooling uses all three archived batches, deduplicating problems by a frozen
content-hash rule. No forecaster or learned admission policy is fitted.

Only runtime-directory placement and its proof checksum may differ when loading
an exported checkpoint. Model revisions, weight checksums, metadata, config
fingerprint, frozen date/template, adapter-file checksums and restored adapter
hash must match. Model downloads use the original pinned official/public source
and exact verified bytes; mismatches fail rather than substitute a model.

The D0/D1 donor-source comparison includes generation-hardware/runtime and
problem/completion differences because D1 is archived and D0 is freshly harvested
on DGX. It is not a pure causal test of harvest quality or model-state effects.
Identical-donor receiver contrasts and recipe contrasts are paired within DGX.
All follow-up parents were previously inspected; this is a paired diagnostic
study, not sixteen fresh independent replications. Original confirmation
endpoints and results remain unchanged.

## Submitted dependencies and transfer handoff

| Stage | Job | State at latest check |
| --- | --- | --- |
| Full CPU suite |28796| Completed;223 tests and lint passed |
| Import and model verification |28797| Held until archive transfer verifies |
| Four-stream real CUDA/adapter/verifier/timing check |28798| Depends on import |
| Saved-data analysis |28803| Completed on the verified outcome registry |
| Matched controls and recipe arms |28800, array0–15%1| Depends on compatibility |
| Complete-cohort paired inference |28801| Depends on all16 array elements |

Original saved-analysis job28799 was cancelled while unstarted;28803 replaced
its unnecessary dependency on the large model/adapter import. No GPU job was
cancelled or replaced for scientific outcomes. Real compatibility measures all
17 final-score costs plus training and conservative harvest allowances before
allowing the48h parent reservations to run. Jobs never shrink scientific budgets
to fit a runtime. Failures and partial outputs remain recorded.

The archive is transferring from Vista through the local machine. A separate
hidden local helper monitors completion, checks the archive SHA256, uploads it
with the DGX SSH key and releases only held import28797. DGX import verifies the
archive again inside its CPU allocation, checks all original files and model
weights, and writes the source-bound readiness receipt. Until this transfer is
complete, the local machine and its Vista transfer connection must stay running;
the submitted Slurm dependency chain does not itself transfer data from this PC.
After verified handoff, Slurm drives import, compatibility, collection and final
analysis independently. Handoff progress is in
`runs/deployment/dgx_transfer_20261026/handoff_status.json`.

Qwen3-8B and Llama-3.1-8B remain later DGX scale extensions. No8B job is part of
this control launch. The SmolLM cohort stays on Vista; its settings and jobs are
unchanged. The original proposal's control budget numbers are historical Vista
estimates; these authorized diagnostic blocks now run on DGX instead.

## First completed DGX analysis

Saved-data analysis28803 completed on all32 original parents. Exploratory joint
parent/shared-item bootstrap intervals, in percentage points:

| Quantity | Mean |95% interval |
| --- | ---: | --- |
| Development-selected H1 |0.016|[-1.286,1.306]|
| Development-selected H2 |-1.760|[-2.983,-0.446]|
| Uniform-candidate H1 |-0.686|[-1.891,0.529]|
| Uniform-candidate H2 |-2.059|[-3.299,-0.797]|
| Selected H2 minus H1 |-1.776|[-2.538,-1.000]|
| Uniform H2 minus H1 |-1.373|[-1.867,-0.871]|

The continued loss remains visible with uniform candidate weighting; selection
alone does not account for the observed mean decrement in this cohort. This does
not identify a mechanism, establish equivalence at H1, or establish robustness
to the update recipe. The original primary and multiplicity-adjusted intervals
remain authoritative for their declared tests. These additional analyses are
exploratory; new paired controls remain pending.

The selected head means (H1, H2, H2-H1) are target
(+1.160,-2.433,-3.593pp), OOD(-1.123,-2.669,-1.546pp), and retention
(+0.012,-0.177,-0.189pp). Target and OOD H2 changes sum to about-5.102pp;
their two-head average is about-2.551pp, not an individual5.1pp collapse.
Eight of32 selected parent means changed from observed positive H1 to negative
H2; six of32 did so with uniform candidate weighting. These observed signs do
not establish latent individual effects. Local report and completion checksums
were verified after download into
`runs/deployment/evidence/dgx_revision_20261026/`.
