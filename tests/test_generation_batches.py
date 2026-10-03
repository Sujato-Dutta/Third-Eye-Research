from dataclasses import replace

import pytest

from third_eye.config import Config
from third_eye.io import digest
from verify_backend import make_tiny_backend


def test_single_prompt_config_hash_compatibility():
    cfg = Config()
    old = cfg.to_dict()
    assert "generation_batch_size" not in old["protocol"]
    assert "sampled_batch_size" not in old["protocol"]
    assert "multiple_choice_scoring" not in old["protocol"]
    assert digest(old) == cfg.fingerprint
    batched = replace(cfg, protocol=replace(cfg.protocol, generation_batch_size=3))
    assert batched.fingerprint != cfg.fingerprint
    assert batched.to_dict()["protocol"]["generation_batch_size"] == 3
    scored = replace(
        cfg,
        protocol=replace(
            cfg.protocol, multiple_choice_scoring="conditional_likelihood"
        ),
    )
    assert scored.fingerprint != cfg.fingerprint


@pytest.mark.integration
def test_independently_seeded_sampling_matches_serial_and_reordered_rows():
    backend = make_tiny_backend()
    prompts, seeds = ["one", "one plus two", "two plus two"], [30, 31, 32]
    before, padding = backend.state_hash(), backend.tokenizer.padding_side
    expected = [
        backend.generate(p, 8, seed=s, temperature=0.7) for p, s in zip(prompts, seeds)
    ]
    backend.config = replace(
        backend.config, protocol=replace(backend.config.protocol, sampled_batch_size=3)
    )
    assert backend.generate_sampled_many(prompts, 8, seeds, 0.7) == expected
    assert backend.generate_sampled_many(prompts, 8, seeds, 0.7) == expected
    assert (
        backend.generate_sampled_many(prompts[::-1], 8, seeds[::-1], 0.7)
        == expected[::-1]
    )
    assert backend.tokenizer.padding_side == padding
    assert backend.state_hash() == before
    with pytest.raises(ValueError, match="seeds"):
        backend.generate_sampled_many(prompts, 8, [30], 0.7)


@pytest.mark.integration
def test_real_greedy_batches_restore_padding_and_preserve_adapter():
    backend = make_tiny_backend()
    prompts = ["one", "one plus two", "two plus two"]
    before = backend.state_hash()
    padding = backend.tokenizer.padding_side
    expected = backend.generate_many(prompts, 3, [1, 2, 3])
    backend.config = replace(
        backend.config,
        protocol=replace(backend.config.protocol, generation_batch_size=2),
    )
    actual = backend.generate_many(prompts, 3, [1, 2, 3])
    assert actual == expected
    assert backend.generate_many(prompts, 3, [1, 2, 3]) == actual
    assert backend.tokenizer.padding_side == padding
    assert backend.state_hash() == before
    with pytest.raises(ValueError, match="seeds"):
        backend.generate_many(prompts, 3, [1])


@pytest.mark.integration
def test_choice_likelihood_matches_independent_full_logits():
    import torch
    from third_eye.training.tokenization import format_prompt

    backend = make_tiny_backend()
    backend.tokenizer.add_tokens(list("ABCD"))
    backend.model.resize_token_embeddings(len(backend.tokenizer))
    backend.config = replace(
        backend.config,
        protocol=replace(
            backend.config.protocol,
            generation_batch_size=2,
            multiple_choice_scoring="conditional_likelihood",
        ),
    )
    prompts = ["one", "one plus two", "two plus two"]
    before = backend.state_hash()
    padding = backend.tokenizer.padding_side
    expected = []
    for prompt in prompts:
        prefix = format_prompt(
            backend.tokenizer, prompt, backend.config.model.chat_kwargs
        )
        scores = []
        for choice in "ABCD":
            inputs = backend.tokenizer(
                prefix + choice,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
                return_token_type_ids=False,
            )
            offsets = inputs.pop("offset_mapping")[0]
            with torch.inference_mode():
                logits = (
                    backend.model(**inputs, use_cache=False)
                    .logits[0, :-1]
                    .float()
                    .log_softmax(-1)
                )
            suffix = offsets[1:, 1] > len(prefix)
            scores.append(
                logits.gather(-1, inputs.input_ids[0, 1:, None])[suffix].sum().item()
            )
        expected.append("ABCD"[max(range(4), key=lambda i: scores[i])])
    assert backend.predict_choices(prompts) == expected
    assert backend.predict_choices(prompts) == expected
    assert backend.tokenizer.padding_side == padding
    assert backend.state_hash() == before
    with pytest.raises(ValueError, match="context"):
        backend.predict_choices(["one " * 130])


def test_choice_scoring_dispatches_only_multiple_choice_examples():
    from types import SimpleNamespace
    from third_eye.data.schema import Example
    from third_eye.evaluation.runner import generate_greedy

    calls = []
    backend = SimpleNamespace(
        config=replace(
            Config(),
            protocol=replace(
                Config().protocol, multiple_choice_scoring="conditional_likelihood"
            ),
        ),
        predict_choices=lambda prompts: calls.append("choices") or ["A"] * len(prompts),
        generate=lambda *a, **k: calls.append("generate") or "answer",
    )
    mcq = Example(
        "m", "one", "A", "retention_dev", metadata={"verifier": "multiple_choice"}
    )
    math = Example("n", "two", "2", "target_dev")
    assert generate_greedy(backend, [mcq], 10, [42]) == ["A"]
    assert generate_greedy(backend, [math], 10, [42]) == ["answer"]
    assert calls == ["choices", "generate"]
