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

## November 11 progress

The first two Llama3B paired trajectories completed, including both policies
through T=5 and the frozen T=3/T=5 scoring points. Jobs `35069` and `35071`
took 14:10:25 and 23:09:36 respectively; their dependent artifact-inspection
and lossless-archive jobs `35070` and `35072` also completed successfully.
The third trajectory, `35073`, is running. The remaining trajectory and 8B jobs
remain queued under the account resource limits. Four-model inference awaits
the fixed cohorts; these execution completions do not establish an efficacy claim.

The storage watcher measured 40,066,170,880 bytes in the additional account,
below its 50-billion-byte cap. The original-account confirmation chain was
released again after a successful repeated storage check, preserving the exact
array elements on which the later Qwen4B extensions depend. No seeds, policies,
scientific sources, or trajectory budgets were changed.

## November 12 progress

Three of sixteen paired trajectories are complete. The third Llama3B trajectory
(`35073`) finished in 16:02:37, and archive/inspection job `35074` completed
successfully. The fourth Llama3B trajectory (`35075`) is running on `dgxa`;
seven of its eight scored checkpoints are recorded. The eight 8B trajectories
and four Qwen4B extensions remain queued. The account watcher reports
40,542,445,568 bytes used, below the 50-billion-byte policy limit. Model and
integrated statistical reports remain dependent on their complete cohorts.

Later on November 12, all four Llama3B trajectories and their archive jobs had
completed; Qwen8B index 0 (`35078`) was running. Pending Qwen8B indices 2/3 and
Llama8B indices 2/3 were delegated to the temporary two-GPU account. Their
original archive jobs now wait for owning-account handoffs. Model and final
review dependencies remain connected. The same fixed seeds, T=3/T=5 scoring,
policies and optimizer budgets are preserved. See
[temporary-worker execution](dgx_temporary_worker_20261112.md) for the exact
dispatch map, storage safeguards and automatic post-lease fallback.

## November 12, 17:39 IST check

All four Llama3B paired trajectories, artifact checks and model review `35077`
are complete. The saved review reports equal-role utility changes relative to
the unchanged parent of -0.212 pp after the first update, -7.113 pp at T=3 and
-16.080 pp at T=5 under the native policy. The T=5 simultaneous interval is
[-20.073, -12.288] pp. Half-LR continuation improves T=5 utility by +6.703 pp
relative to native continuation (simultaneous interval [3.222, 9.784] pp);
this is not a positive gain relative to holding the parent. These are the
exploratory four-seed Llama3B math results, not a completed cross-model finding.

Qwen8B trajectory `35078` remains running at 6:15 elapsed. The main confirmation
remains at 3/20 complete, with `35053_3` running at 12:00 elapsed. The temporary
GPU preflight still awaits allocation. Slurm's current estimated start is
November 13, 19:23 IST; that estimate can change. All ten delegated jobs remain
connected behind preflight, and the three autonomous safeguards are running.
