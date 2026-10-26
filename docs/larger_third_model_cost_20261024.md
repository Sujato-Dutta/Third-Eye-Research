# Larger third-family model: planning estimate

Assessment only, October24. No new experiments submitted. The SmolLM proposal
remains unchanged pending a model/scope decision.

Current confirmation: Qwen3-4B and Llama-3.2-3B-Instruct. SmolLM3-3B is proposed.
Qwen3-8B or Llama-3.1-8B would add scale without a third independent family.
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
could add both; its official card provides Apache-2.0 weights and Transformers
usage. Model-specific masking and training compatibility remain to be tested.

Completed32 parent allocations used308.1719 GPU SUs, or9.6304 per parent.
Sixteen similarly distributed small-model parents cost154.0860 SUs.
A **hypothetical1.5--2.5x slowdown**, not a measured7B result, gives231--385 SUs
for16 larger-model parents before new-model setup or failures. Initial planning
ranges are **250--400 SUs for16 runs** or **125--210 for8**. These are not
worst-case bounds. An8B model may cost more. Generation length, correction
supply and batching prevent a guarantee from parameter count alone.

Retaining the current controls, reporting and contingency uses308 SUs outside
the third-model block. Replacing192 reserved SmolLM SUs with250--400 larger-model
SUs gives558--708 additional SUs, exceeding500 as currently reserved.

An alternative within500 is8 Mistral parents,4 math and4 code, keeping K=3,
three continuations/candidate and128/128/256 scoring:

| Block | Maximum SUs |
| --- | ---: |
| CPU analysis, GPU checks and reporting | 14 |
| Existing-model recipient/data controls | 128 |
| Existing-model recipe controls | 96 |
| Larger third model,8 parents | 200 |
| Failure/runtime contingency | 62 |
| **Additional ceiling** | **500** |

The200-SU reservation is an allocation, not a verified cost bound. Measure
full-task runtime coverage before bulk submission, using contingency if needed
within500. Do not silently reduce sample sizes, repeats or scoring sets.
Eight runs offer less third-model precision than16. Neither design establishes
a causal model-size effect or universal model-family conclusions.

Only attention LoRA adapters are trained. Current rank16 q/k/v/o adapters have
approximately9.175M parameters for Llama3.2-3B and11.796M for Qwen3-4B, derived
from their configurations; these differ from total3B/4B base-model parameters.
The original recipe uses50 AdamW steps, LR1e-4, two correction examples and
gradient accumulation8.
