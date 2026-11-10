# Longer-horizon execution — November 9, 2026

The [frozen design](long_horizon_20261109.md) contains four paired seeds per
backbone, sixteen total. Each trajectory supplies both T=3 and T=5 results.
The existing twenty-parent two-update study continues on the original account.
No seed is removed from that primary study.

## Submitted chain

| Work | Slurm jobs | Dependency |
| --- | --- | --- |
| Pinned model staging on CPUs | 35062 | Completed |
| Full source validation | 35063 | Completed: 268 tests and lint passed |
| Original input binding | 35064 | Failed on input-file read permissions; retained |
| Read-access repair | 35107 | Completed on original-account CPU allocation |
| Repeated binding and model verification | 35108 | Completed |
| Four real CUDA/model preflights | 35065–35068 | All completed successfully |
| Llama-3.2-3B trajectories | 35069, 35071, 35073, 35075 | All preflights pass |
| Qwen3-8B trajectories | 35078, 35080, 35082, 35084 | All preflights pass |
| Llama-3.1-8B trajectories | 35087, 35089, 35091, 35093 | All preflights pass |
| Qwen3-4B extensions | 35096, 35098, 35100, 35102 | Preflights and corresponding primary parent |
| Per-model statistical reports | 35077, 35086, 35095, 35104 | Four inspected, archived trajectories/model |
| Integrated extension report | 35105 | All four model reports |

Every trajectory has its own dependent CPU archive/inspection job. This releases
the GPU while retention work runs, and allows useful CPU work to overlap the
next trajectory. The account permits one GPU and 32 CPUs/256 GB RAM in aggregate;
Slurm enforces these limits across GPU, verification, archive and monitor jobs.
The first account's independent GPU continues the primary study concurrently.
The first additional-account trajectory, Llama3B seed `2026111000` (`35069`),
has started on `dgxb` after all four preflights passed. These checks established
real training, exact rollback, batched generation and conditional-choice scoring;
they are not scientific efficacy results.

Qwen extensions wait only for their corresponding primary array elements
`35051_0`, `35051_1`, `35053_2` and `35053_3`. The canonical source draw is always
zero, fixed before outcomes. No completed update is retrained to obtain a second
horizon. Array-element dependencies follow the
[Slurm dependency specification](https://slurm.schedmd.com/job_array.html#depend).

## Validation, access and storage

Scientific source:
`adb48501b796b2c20538a0e0142379b13f9f3f659338f6bff18c4200ebd72d2a`.
Source archive:
`1c9ae90d43907bf5e442836ac931d7828e0fdddd9f4e661f1f3b8b606cec388a`.

Model staging verified all pinned Llama3B, Qwen8B and Llama8B shards against the
previous official-byte catalogue/registry. Read-only reuse avoids duplicate
Qwen4 weights and another environment. Whole-account usage after staging was
**38,902,317,056 bytes**. The 50-billion-byte limit, 48-billion-byte admission
threshold and two-GB per-job scratch reservation remain active.

Input binding initially found owner-only JSON files in the original frozen
benchmark/config directories. The first ACL helper (`35106`) found that the ACL
utility was unavailable and changed no data. Replacement `35107` made only the
owned public benchmark/config JSON files readable. No credentials, data content,
scientific hashes, recipe or seed changed. Pending preflight dependencies were
updated to successful binding `35108`; the original failed jobs and submission
receipt remain preserved. `dependency_repair.json` records the scheduler repair.

The new account has a 450-GPU-hour protective ceiling for the extension: 448 for
the sixteen trajectory jobs and two for compatibility. It is a maximum, not an
ETA. The project targets November 29 assuming continuous GPU access; scheduler
delays and actual primary-study timings still affect calendar completion.

Remote campaign: `/dgxa_home/se23ucse205/third_eye_horizon_20261109`.
Local release and receipts: `runs/deployment/long_horizon_20261109/`.
Storage-watch jobs `35061 → 35109 → 35110` cover the execution window in
successive one-CPU allocations, checking every twenty minutes. Per-job admission
checks remain mandatory; the watcher is not an enforced filesystem quota.
The original manuscript and Overleaf archive remain unchanged until results are
available and reviewed. Four seeds per backbone are a sensitivity design, not
a guarantee of precise individual-model effects.
