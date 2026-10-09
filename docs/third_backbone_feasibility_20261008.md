# Third-backbone feasibility, October 8

Assessment only. No third-model experiment has been queued, and the frozen E2
cohort, primary comparison and scientific budgets are unchanged.

A third independently trained backbone would broaden the tested population.
It cannot substitute for matched reliability controls or establish that ranking
changes exceed measured variation. The deadline priority remains complete
balanced confirmation and reliability evidence on the two existing backbones.

## Model options

| Candidate | Evidence and integration | Deadline assessment |
| --- | --- | --- |
| HuggingFaceTB/SmolLM3-3B | Public Apache-2.0 3B text model; its official card places support in Transformers4.53.0, below our pinned4.56.2. Standard causal-LM loading and an explicit thinking toggle are documented. Real LoRA, masking, batching, checkpoint and continuation tests are still required. | Lowest apparent integration risk; compatibility is inferred, not yet validated on Vista |
| google/gemma-3-4b-it | Existing frozen backend has a Gemma3 composite loader and limits LoRA to language layers. Hugging Face requires accepted access conditions. No Gemma directory was found in the checked Vista model cache. | Reasonable if approved access is readily available; model-specific validation and revision pinning remain necessary |
| google/gemma-4-E2B-it | Official small Gemma4 variant:2.3B effective parameters,5.1B including embeddings. Public Apache-2.0 weights; the family is absent from our older pinned runtime and existing special loader. | Needs a separate newer environment and loader/adapter tests; greater deadline risk |

Sources: [SmolLM3 official model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B),
[Gemma3 official access conditions](https://huggingface.co/google/gemma-3-4b-it),
[Google Gemma4 specifications](https://ai.google.dev/gemma/docs/core/model_card_4),
[official Gemma4 Transformers implementation](https://huggingface.co/docs/transformers/en/model_doc/gemma4).

## Scope and planning costs

A balanced full supplement uses two seeds across mathematics and code: four T5
trajectories, potentially20 states/60 candidate labels, plus four independent
early/late reliability parents. Existing-backbone measurements extrapolate to
roughly110–150 additional GPU node-hours with controls, and approximately32–58
hours on the critical path after suitable allocations. These are planning
estimates, not measured runtimes for a new backbone. Downloads, validation,
model-specific generation speed, scarcity and queue waits add uncertainty.

A bounded breadth check instead uses four single-state runs, two seeds per task,
with unchanged K=3 and H=2 labeling. Estimate15–30 GPU node-hours before controls;
four matched-control parents add approximately27–47. This supports additional
two-step cross-backbone evidence only. It does not justify claiming a completed
five-generation third-backbone study. Preserve and report incomplete states.

The new backbone must remain a separately declared external confirmation set.
Do not change the existing E2 primary population, select methods using its
outcomes, tune on the new labels or merge uneven coverage into a balanced claim.
Freeze model identity/revision, seeds, scope and all analysis choices before
outcomes, following a compatibility-only preflight.

## Protecting the current queue

As of15:24 UTC, core CUDA validation1057730 still awaits allocation. All eight
E2 trajectories remain dependency-blocked; the three original recoveries are
queued for Priority. No new Gemma weights were found in the checked model cache,
and this check did not return a current service-unit balance.

The existing package can occupy19 GPU jobs, including controls submitted later.
Additional pending jobs count toward the20-job relay cap. Therefore new work
cannot simply occupy all apparently empty slots: it must reserve capacity for
the unscheduled core controls and use capacity released by core jobs. A naive
four-trajectory/eight-job expansion could otherwise cause core callback rejection.

Recommendation: explore the public3B text candidate in a bounded supplement if
spare capacity and the model-specific preflight support the deadline. Consider
Gemma3 if access is immediately available. Do not make a new Gemma4 runtime or
a full third-family campaign a prerequisite for submission. GPU size itself is
not the main uncertainty; timely allocation and a validated full pipeline are.
