# Third Eye novelty audit — October 15, 2026

This is a literature and evidence review, not a declaration of a new experiment.
The broad reliable-evaluation framing is insufficiently distinctive by itself.
Third Eye still has a plausible, narrower contribution in the interaction between
persistent parameter changes and the verified correction data those changes
generate for the next update. That mechanism and its practical importance remain
unproven. Neither a forecasting advantage nor an ACL Main/Oral result is established.

## Scope and confidence

Targeted searches covered fine-tuning performance prediction, predictive agent
self-modification, self-editing, sequential model editing, forgetting forecasts,
short-horizon meta-optimization, recursive synthetic data, performative prediction,
selection bias, experimental variance and budget-aware evaluation. Searches used
paper titles and combinations of persistent parameter updates, continuation,
ranking, self-generated corrections, noise, selection regret and data/parameter
controls. Closest papers were followed into related work and experimental sections.
Primary sources below include proceedings, ACL Anthology and author preprints.

Full-text checks were especially important for TuneAhead's probe/PEFT conditions,
SEAL's sequential self-edits, Predict Before You Modify's fixed LLM weights,
AIDE²'s nested-noise discussion and Nakamura et al.'s corpus controls. Several
supporting comparisons rely on publisher abstracts rather than a complete method
audit. This is a targeted novelty audit, not an exhaustive systematic review.
Failure to find an exact match does not establish priority. Preprints are labeled
as such; their reported results are not independently reproduced here.

## Closest overlaps

| Prior work | What it already covers | Remaining distinction worth testing |
| --- | --- | --- |
| [TuneAhead (2026)](https://arxiv.org/html/2606.17660v1) | Predicts fine-tuning outcomes from dataset descriptors and short training probes; discusses LoRA, screening and diagnostic features. | Value after another endogenous self-update, rather than the outcome of an isolated candidate run. A short probe plus a predictor is not itself new. |
| [Predict Before You Modify (2026 preprint)](https://arxiv.org/html/2610.13991v1) | An Improvement Dynamics Model forecasts proposed agent modifications and guides revisions; predictive knowledge accumulates across generations. Appendix A.3 explicitly keeps underlying LLM weights fixed. | Persistent weight updates and their effects on the next correction generator. Broad claims of the first predictive self-evolution system would be inappropriate. |
| [AgentSquare (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0ae94013da7cd459402fd77874e09ee3-Abstract-Conference.html) | Uses an in-context surrogate to predict performance and skip unpromising agent designs. | Forecasting recursive parameter-update consequences, rather than modular agent-design performance. Surrogate-assisted candidate screening is established. |
| [SEAL: Self-Adapting Language Models (2025)](https://arxiv.org/html/2506.10943v1) | Generates training data and update directives for persistent weight changes. Its limitations include an explicit sequential-self-edit forgetting experiment. | Independently scored H1/H2 candidate comparisons with controlled future correction data. Persistence, self-generated training data and sequential interference are already covered. |
| [What Will My Model Forget? (ICML 2024)](https://proceedings.mlr.press/v235/jin24d.html) | Forecasts examples forgotten after language-model refinement and supports targeted mitigation. | Future value after a second self-generated update. Predicting retention side effects of corrections is not a new target by itself. |
| [Demystifying Language Model Forgetting with Low-rank Example Associations (2024 preprint)](https://arxiv.org/abs/2406.14026) | Models structured associations between refinement tasks and forgotten examples, including prediction through low-rank structure. | A checkpoint-by-correction-batch interaction tied to recursive selection decisions. A matrix representation or low-rank predictor alone would not distinguish Third Eye. |
| [Reinforced Lifelong Editing (ICML 2025)](https://proceedings.mlr.press/v267/li25an.html) | Learns sequential editing behavior with rewards over a knowledge-edit sequence and changing parameter states. | Endogenous verified corrections and noise-controlled candidate selection. Optimizing beyond one edit is established. |
| [More Edits, More Stable (ICML 2026)](https://proceedings.mlr.press/v306/ma26f.html) | Explains how earlier edits influence later editing stability through lifelong normalization. | Mechanisms specific to self-generated LoRA corrections. The idea that updates alter future adaptability is already studied. |
| [Understanding Short-Horizon Bias in Stochastic Meta-Optimization (2018)](https://arxiv.org/abs/1803.02021) | Analyzes how short-horizon optimization can favor decisions that underperform over longer horizons. | Whether endogenous correction feedback produces a measurable, independently scored selection effect in this protocol. Horizon mismatch alone is not a new principle. |
| [Recursive self-improvement of AI research agents / AIDE² (2026 preprint)](https://arxiv.org/html/2609.26457v1) | Studies recursive research-agent code improvement and explicitly discusses nested evaluation noise, false acceptance and expensive replication. | Quantitative controls for persistent parameter-update selection. A general warning that noisy evaluations can derail RSI is already present. |
| [Demystifying Mixed Outcomes of Self-Training (Findings EACL 2026)](https://aclanthology.org/2026.findings-eacl.213.pdf) | Studies recursive synthetic pre-training. Section 4.1 changes real versus synthetic corpora while matching training settings to investigate data quality versus optimization explanations. | Crossing several nearby recipient checkpoints with their own verified correction batches to explain candidate ranking. Disentangling data quality from training settings broadly is not a first claim. |
| [Performative Prediction (ICML 2020)](https://proceedings.mlr.press/v119/perdomo20a.html) | Formalizes feedback in which a model changes the distribution encountered during retraining. | A concrete intervention and decision consequence for verified self-correction. Endogenous distribution feedback is an established concept. |

The collision with Predict Before You Modify is particularly important for the
original forecasting narrative. Its setting differs, so it does not eliminate
every Third Eye contribution. It does eliminate an unqualified claim that
forecasting proposed self-modifications using an improvement-dynamics model is
new. Its displayed submission date and arXiv month identifier are inconsistent;
verify version history before making precise chronology or priority arguments.

## Statistical rigor is necessary, but is not the central novelty

| Proposed ingredient | Established reference | Implication |
| --- | --- | --- |
| Separate selection from scoring to reduce optimism | [Cawley and Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html) | Winner selection from noisy criteria and resulting evaluation bias are established problems. |
| Protect final evidence from repeated adaptive tuning | [Preserving Statistical Validity in Adaptive Data Analysis](https://arxiv.org/abs/1411.2664) | Fresh confirmation is sound practice; our design does not inherit that paper's formal adaptive-data guarantees. |
| Replicate important sources of experimental variation | [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098) | Multiple seeds and variance decomposition need a setting-specific discovery or useful decision result. |
| Report uncertainty with few independent runs | [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) | A stronger protocol contribution should change substantive conclusions and be useful beyond one small cohort. |
| Compare quality at matched computational budgets | [Show Your Work, EMNLP 2019](https://aclanthology.org/D19-1224/) | Compute-aware comparisons and reporting search effort are already established in NLP. |
| Make LLM evaluation cheaper and less variable | [tinyBenchmarks](https://arxiv.org/abs/2402.14992), [Fluid Language Model Benchmarking](https://arxiv.org/abs/2509.11106) | Item-efficient evaluation is prior art; continuation-branch uncertainty is a distinct additional problem. |
| Fall back to a baseline under uncertainty | [Safe Policy Improvement with Baseline Bootstrapping](https://arxiv.org/abs/1712.06924) | Abstention is not a new general principle. Offline-RL guarantees do not automatically transfer to our weight-update selector. |

## What the current evidence can carry

The completed [evidence review](evidence_review_20261011.md) reports a frozen H2
versus H1 selection effect near zero on the fresh 40-state cohort, with a 95%
interval of approximately [-0.60,+0.63] utility percentage points. Independently
scored observed-future versus immediate selection is +0.34pp [-0.55,+1.24]. Raw
H1/H2 winner disagreement has not been established as larger than measured fresh
continuation variation. These results do not establish a repeat-stable horizon
penalty, intrinsic unpredictability or a useful learned selector.

The observed-future selector's secondary +1.61pp [+0.88,+2.34] advantage over random
shows a possible signal worth understanding. It requires privileged future
observations and does not demonstrate an accessible forecasting advantage.

The [October 13 development review](forecasting_review_20261013.md) does not restore
the expected-value forecaster's advantage. The [October 14 LR intervention](mechanism_review_20261014.md)
successfully reduced measured adapter-change norms, but its primary selected-
candidate effect is -0.24pp [-0.97,+0.37]. A nominally positive all-candidate
secondary result did not survive the declared multiplicity adjustment. These
eight previously inspected parents remain development evidence.

Therefore, another architecture swap, a new title, confidence intervals, or more
of the same LR sweep cannot be treated as a demonstrated novelty upgrade.

## Most promising specific question

An update changes both the model that solves tasks and the model that generates
the next training batch. Does apparent future-value instability come from the
recipient parameter state, the correction batch it generates, or the interaction
between them? Does that distinction change which update should be selected?

This is a narrower hypothesis than generic data feedback or catastrophic
forgetting. I did not find the exact crossed experiment below in the closest
papers inspected. That is a provisional gap, not proof of a first contribution.
The crossed design itself is a standard factorial control; its potential value
is the new empirical explanation and resulting selection consequence.

### Proposed diagnostic, not yet implemented or submitted

For each parent, retain its three existing candidate checkpoints M1_i. For a
matched continuation draw s, retain the verified correction batch B_j_s generated
by candidate j. Evaluate all recipient/donor combinations within the same parent,
backbone and task stream:

    Y(i,j,s) = U(Update(M1_i, B_j_s)) - U(parent)

The existing endogenous experiment observes the diagonal i=j. The off-diagonal
cells apply another candidate's saved batch to a recipient checkpoint. Report
both total future utility and continuation gain relative to that recipient's
own M1 score: otherwise a row effect may simply reflect an immediate-score gap.

| Recipient | Batch generated by candidate 0 | Batch generated by candidate 1 | Batch generated by candidate 2 |
| --- | --- | --- | --- |
| M1_0 | Existing native continuation | New crossed continuation | New crossed continuation |
| M1_1 | New crossed continuation | Existing native continuation | New crossed continuation |
| M1_2 | New crossed continuation | New crossed continuation | Existing native continuation |

Use all donors uniformly rather than selecting a favorable reference batch.
Match optimizer seed, steps, LR, rank, input ordering and evaluation manifests
within each draw; preserve checkpoint and batch hashes. A draw bundles generation
and optimizer randomness unless those are explicitly separated. With three
draws, the full design is 27 branches per parent: nine native and 18 additional
off-diagonal branches. Eight development parents would require 144 additional
training/evaluation branches. This is not a claim of 144 independent samples.

If checkpoints and complete batch artifacts are reusable, new harvesting is
unnecessary. Saved-batch availability, identical training semantics and numerical
replay still need verification. Branch count alone does not determine SUs or wall
time; use measured training/evaluation timings and include validation/retries
before reserving any part of the existing 1,200-SU cap.

Important limits:

- A donor correction need not be a failure for its recipient. Off-diagonal cells
  are deliberately counterfactual controls, not normal endogenous self-updates.
- Effects are conditional on these checkpoints and generated batches. The design
  is not automatically a natural-direct/indirect-effect mediation analysis.
- Batch content can be held fixed, but token losses, gradients and their
  interaction with recipient parameters appropriately remain different.
- Terminal branches require their own declared handling; no fake batches or
  duplicated examples may be introduced to complete a matrix.
- Current eight-parent analysis would be exploratory. Choose a primary contrast
  and practical effect size before collecting fresh balanced confirmation.
- Native-versus-crossed ranking differences must be independently scored and
  compared with appropriate continuation variation. A significant cell contrast
  alone does not prove useful selection regret or forecastability.

This control could reveal a stable interaction, a primarily data-driven effect,
a primarily recipient-state effect, or no resolvable effect. Only the first three
with adequate precision would motivate a specific mechanism story; each still
needs a clear comparison with the prior work above.

## What would make the contribution persuasive

1. Establish a practically meaningful, reproducible effect specific to endogenous
   correction feedback, with balanced fresh parents and independent scoring.
2. Explain a decision consequence: when does immediate selection choose the
   wrong update, and how much independently scored value is at stake?
3. Derive and freeze one accessible diagnostic or selector from development
   findings. Compare with immediate gain, simple feature models, a properly
   documented TuneAhead-style adaptation, random and no-update at matched cost.
   Post-update or future-only measurements must be charged and labeled.
4. Confirm useful performance on untouched trajectories. An architectural paper
   needs a learned method that beats credible simple alternatives; a mechanism
   paper can succeed without one if the explanation is robust and consequential.
5. Expand backbone/scale coverage only after a substantive effect is established.
   Four early states of a third model remain a sanity check.

Do not automatically turn any significant subgroup, correlation or secondary
endpoint into the headline. If the crossed control is inconclusive or merely
reproduces familiar forgetting/data-quality effects without a new decision
consequence, do not launch a large forecasting sweep on that basis. Preserve the
negative findings and reconsider the narrower measurement-paper scope.

## Recommendation

My literature-based assessment is that reliable evaluation alone offers limited
conceptual novelty, while a demonstrated parameter/data-feedback mechanism could
support a more distinctive empirical paper. A useful forecast based on that
mechanism would strengthen the method story, but cannot be assumed in advance.

A conditional working title is **Third Eye: Disentangling Parameter and Data
Feedback in Recursive Self-Updates**. It describes a proposed investigation,
not a completed finding. The current contribution remains an evaluation study
with unresolved mechanism and selection benefits. Main/Oral competitiveness
depends on what fresh evidence establishes, not on adopting this title.
