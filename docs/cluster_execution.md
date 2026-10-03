# DGX campaign status

The active research campaign migrated to [TACC Vista](vista_execution.md) on
October 1. DGX jobs `25240`, `25241`, `25244`, and `25245` were retired; `25239`
completed its pilot. The record below describes the archived DGX deployment.
Its results are calibration evidence and are excluded from Vista study labels.

Campaign launched October 1, 2026 under `~/third_eye_research`.
Research outcomes remain pending complete matched-state and forecast artifacts.

| Stage | SLURM job | Status at launch |
| --- | --- | --- |
| Environment setup | 25108 | Complete |
| Official math/code dataset preparation | 25112 | Complete |
| CUDA LoRA/NF4 training and rollback checks | 25127 | Passed |
| Verified public Llama weight/tokenizer source | 25145 | Passed |
| Isolated code verifier and 252 reference programs | 25143 | Passed |
| CPU verification profile | 25235 | Passed |
| Updated 74-test suite including quota regressions | 25243 | Passed |
| CUDA/NF4 and real Qwen batched inference | 25163 | Passed |
| Conditional multiple-choice GPU scoring | 25217 | Passed |
| Earlier Qwen3-4B math pilot | 25218 | Failed: 5 corrections, batch required 32 |
| Full-A100 NF4 generation/sampling profile | 25236 | Passed |
| Full-A100 bf16 generation and LoRA update/rollback | 25237 | Passed |
| Initial full-A100 Qwen3-4B math pilot | 25238 | Failed: infeasible random quota; fixed |
| Full-A100 Qwen3-4B math retry | 25244 | Queued |
| Full-A100 Llama3.2-3B math pilot | 25239 | Running; LoRA updates underway |
| Full-A100 Qwen3-4B code pilot | 25240 | Submitted |
| Full-A100 Llama3.2-3B code pilot | 25241 | Submitted |
| Automatic study continuation | 25245 | Queued after pilots |

Earlier single-prompt and generation-only retention pilots were superseded
before any complete research state was produced. The active protocol fixes
greedy batches at 32 prompts for core models and 16 for transfer models.
Multiple-choice retention ranks A/B/C/D by conditional answer log likelihood;
math/code retain greedy generation. A hardware check verified repeatability,
batched/individual agreement, adapter isolation and padding restoration.
Eight real retention questions took 1.10 seconds and 4.32 GB peak allocated
CUDA memory. This is a scoring check, not a benchmark-accuracy claim.
All earlier GPU time, verification and the failed quota pilot (4.9489 hours before the retained/retry pilots) remain included
in the budget. Superseded pending jobs were cancelled; logs remain diagnostic.

Both nodes were checked: 370 logical CPUs were unallocated at inspection, but
the account permits only 32 CPUs and one GPU simultaneously. A free full
A100-SXM4-40GB on `dgxb` was used for scheduled profiles. Research jobs now
request one full A100, 16 CPUs and 64 GB host memory. Bf16 LoRA avoids NF4
overhead on this allocation. Sampled corrections use independent per-example
random streams with fixed batches of 16 for core models and eight for transfer.
Isolated code verification uses 16 workers; symbolic verification stays serial.

| Workload | Earlier setting | Active setting | Measured ratio |
| --- | --- | --- | --- |
| 32 greedy prompts, 128-token limit | NF4, batch 8: 52.51 s | bf16, batch 32: 7.79 s | 6.7x |
| Eight sampled prompts, 64-token limit | NF4, serial: 44.34 s | bf16, batched: 3.88 s | 11.4x |
| 252 canonical code references | One worker: 19.17 s | 16 workers: 3.26 s | 5.9x |

These are workload profiles, not end-to-end study speedups or accuracy gains.
Precision and batching can change generated tokens; each setting enters the
protocol fingerprint and stays fixed across compared policies. The bf16 probe
also passed a real LoRA update and exact adapter rollback, using 9.09 GB peak
training memory on its two-example, one-step check. Full pilot states still
determine usable correction yield, maximum sequence length and actual runtime.

The subsequent Qwen pilot failed before any matched state was published:
49 initial errors yielded only five verified corrections from 182 retry
attempts. Three distinct difficulty-matched batches of size two are feasible
from that saved pool. The new pilots fix this size uniformly across all five
backbone templates, both task families and all candidates, preserving K=3
and H=2. Calibration uses training-pool yield, not held-out gains or gate
outcomes. Other pilot jobs were superseded to maintain protocol consistency.
Pilot allocations allow up to 12 hours to measure a complete H=2 state;
measured runtime must still satisfy the campaign's compute and task ceilings.
Small correction pools at later checkpoints remain a possible stop condition.

The full-A100 Qwen run reached candidate construction in 11m56s. Its five
verified corrections were split into two short and three long examples. A
random quota selected both short examples, allowing only one unique batch;
the initial sampler incorrectly rejected the whole pool. The fixed sampler
preserves feasible seeded quotas and otherwise finds a matched quota with
maximum distinct-combination capacity using pool counts alone. The exact
failed pool now yields three distinct size-two batches with identical strata.
Regression checks also retain rejection of truly insufficient matched pools.
Only this failed Qwen job was retried; running Llama and queued code jobs were
retained. The failed run remains diagnostic and is excluded from research labels.

Direct original-repository downloads succeeded for Qwen. Llama and Gemma
original download endpoints required authentication. Public mirrors expose
weight shards with the same hashes; the Llama core shards were downloaded and
checked against the original SHA-256 values. Its original tokenizer vocabulary
was recovered from a pinned public copy and verified by Git blob hash.
Mirror configuration differences are preserved in its immutable source proof.
Transfer downloads repeat these checks after the measured core gate passes.

The active manifest is `experiments/campaign_20261001_v6.json` on DGX.
Controller state is written to `runs/campaign_20261001_v6/status.json` and
`events.jsonl` after it starts. It submits the staged core collection,
forecasters/ablations, gate analysis, policy comparisons, final evaluation,
all three conditional transfer backbones, reports and budget-permitted stress
tests. It stops on a failed prerequisite, insufficient/failed gate, task failure,
or exhausted compute/storage budget. Submitted jobs remain visible through
`squeue`/`sacct`; a configured downstream stage is not a submitted GPU job until
its prerequisites and reservations pass.

Gate 1 requires at least 20 complete states and the predefined reversal/harm
signal. Gate 2 requires at least 30 informative validation states and the
predefined Direct H=2 ranking thresholds. Infrastructure passes establish
implementation behavior; they do not establish either research gate or an
ACL-level empirical result.

```bash
cd ~/third_eye_research
squeue -u "$USER"
sacct -j 25244,25239,25240,25241,25245 --format=JobID,State,Elapsed,ExitCode
tail -n 30 third_eye_pilot_25239.out
tail -n 30 third_eye_controller_25245.out
cat runs/campaign_20261001_v6/status.json
```

The scheduler reported a group GPU-resource limit while the first pilot was
running, so the remaining pilots wait in the queue. Training stays within
compute-node allocations; the configured ceiling is 160 GPU wall-hours,
including pilot/verification spending, with a 50 GB project storage limit.
