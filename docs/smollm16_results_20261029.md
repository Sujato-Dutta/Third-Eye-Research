# SmolLM3 complete third-family results

October29,2026. All16 parents (eight math/eight code) and analysis1101557 completed successfully. All208 scored checkpoints and144 continuation draws passed the frozen analysis; zero draws were terminal. The report SHA256 was verified after download.

Source: `090fc9a124138c9538b7f0e1d35d38b85ebfa1224328afa021d97d7c1572f8b6`. Plan SHA256: `caf5b6b501a358de24f8c2a3cab53af33a96dd19ae02aef98c53d5942e9f6928`. Report SHA256: `4af26a995f22c9a85b24876b413c62400fb170f32496dac3adfa948764d751ab`. Local evidence: `runs/deployment/evidence/smollm16_20261029/`.

## Declared aggregate endpoints

Utility is the equal-weight mean of target, OOD and retention accuracy. All entries below are percentage points. H1 and H2 are relative to the unmodified parent; change is H2 minus H1. Intervals jointly resample parents within task and shared exact scoring-item identities. Continuation draws stay nested within parent.

| Endpoint | Mean |95% interval |
| --- | ---: | --- |
| selected_h1 | -3.406 | [-5.575, -1.277] |
| selected_h2 | -7.667 | [-10.521, -5.059] |
| selected_change | -4.262 | [-6.163, -2.581] |
| uniform_h1 | -3.651 | [-5.561, -1.622] |
| uniform_h2 | -8.023 | [-10.088, -5.870] |
| uniform_change | -4.372 | [-5.319, -3.473] |

## Role and task decomposition

| Role | Selected H1 | Selected H2 | H2 minus H1 |95% change interval |
| --- | ---: | ---: | ---: | --- |
| ood_test | -5.298 | -11.296 | -5.998 | [-8.724, -3.369] |
| retention_test | +0.085 | -0.240 | -0.326 | [-1.318, +0.655] |
| target_test | -5.005 | -11.466 | -6.462 | [-11.312, -2.588] |

Task-specific selected changes are descriptive:
- smollm3-3b/code: -4.867pp,95% interval [-6.771, -3.109].
- smollm3-3b/math: -3.657pp,95% interval [-7.018, -0.814].

## Interpretation and scope

The predeclared continuation-loss endpoint is negative in a third model family. It also remains negative with uniform candidate weighting, so the development selector alone does not account for the mean decrease. The additional loss is concentrated in target and OOD performance; the retention-change interval crosses zero.

SmolLM already deteriorates at H1: its aggregate H1 interval excludes zero. It therefore extends continuation harm but does not reproduce the near-neutral H1 pattern of the original two-model aggregate. Do not describe immediate harmlessness as universal, use this result as proof of a mechanism, or claim that fewer steps or other recipes fail before the paired controls finish. The original32-parent primary and inference remain unchanged; this is the separately declared16-parent extension.

No forecasting model was trained. The report does not establish scale generality, five-model completion, a causal size effect, or multi-generation policy success. The queued8B checks remain future evidence.

## Completed Vista accounting

Revision usage including checks, failed setup attempts and the input export is **218.1069 SUs**, leaving **281.8931** under the500-additional-SU ceiling. Full reservations released 207.9106 unused SUs. This project allowance differs from the service-reported account balance. DGX extensions consume zero Vista SUs.
