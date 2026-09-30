from dataclasses import replace
import json
import math
import re

import pytest

from third_eye.data.schema import Correction, Example
from third_eye.training.tokenization import collate, encode_completion
from verify_backend import make_tiny_backend, verify


@pytest.mark.integration
def test_real_lora_training_and_checkpoint_roundtrip():
    report = verify()
    assert report["status"] == "passed"
    assert report["training"]["optimizer_steps"] == 3


@pytest.mark.integration
def test_prompt_and_padding_are_excluded_from_loss_and_truncation_rejected():
    b = make_tiny_backend()
    row = encode_completion(b.tokenizer, "one plus two", "#### 3", 64, {})
    assert all(v == -100 for v in row["labels"][:6])
    assert any(v != -100 for v in row["labels"])
    shorter = encode_completion(b.tokenizer, "one", "#### 2", 64, {})
    batch = collate([row, shorter], b.tokenizer.pad_token_id)
    assert all(v == -100 for v in batch["labels"][1, len(shorter["labels"]) :])
    with pytest.raises(ValueError, match="exceeds"):
        encode_completion(b.tokenizer, "one " * 70, "#### 3", 64, {})


@pytest.mark.integration
def test_chat_template_masking_handles_generation_prefix():
    b = make_tiny_backend()
    b.tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'assistant: '}}{% endif %}"
    row = encode_completion(b.tokenizer, "one plus two", "#### 3", 64, {})
    assert -100 in row["labels"] and any(x != -100 for x in row["labels"])


@pytest.mark.integration
def test_checkpoint_corruption_and_config_mismatch_rejected(tmp_path):
    b = make_tiny_backend()
    cp = tmp_path / "checkpoint"
    b.save_checkpoint(cp, {})
    config = b.config
    b.config = replace(config, training=replace(config.training, learning_rate=0.001))
    with pytest.raises(ValueError, match="protocol"):
        b.load_checkpoint(cp)
    b.config = config
    metadata = json.loads((cp / "state.json").read_text())
    weight = cp / next(iter(metadata["weights"]))
    weight.write_bytes(weight.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        b.load_checkpoint(cp)


@pytest.mark.integration
def test_assistant_loss_is_finite_and_seeded_updates_differ():
    b = make_tiny_backend()
    ex = Example("x", "one plus two", "#### 3", "train")
    assert math.isfinite(b.measure_loss([ex]))
    before = b.state_hash()
    b.train([Correction(ex, "#### 3", 1)], seed=4, max_steps=1)
    assert b.state_hash() != before


@pytest.mark.integration
def test_real_training_through_all_h2_branches(tmp_path):
    """Generation is a fixture; every LoRA/probe/checkpoint operation is real."""
    from third_eye.data.schema import ROLES
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.experiments.labeling import LabelGenerator, run_trajectory

    b = make_tiny_backend()
    config = replace(b.config, protocol=replace(b.config.protocol, candidate_size=2))
    b.config = config

    def fixture_generate(prompt, *args, **kwargs):
        value = int(re.search(r"What is (\d+) plus 2", prompt).group(1))
        return f"#### {value + 2}" if "Reconsider" in prompt else "#### -1"

    b.generate = fixture_generate
    splits = {
        role: [
            Example(
                f"{role}-{i}",
                f"What is {offset + i} plus 2?",
                f"#### {offset + i + 2}",
                role,
            )
            for i in range(6 if role == "train" else 2)
        ]
        for role, offset in zip(ROLES, (0, 10, 20, 30))
    }
    initial_hash = b.state_hash()
    labeler = LabelGenerator(
        b, config, splits, VerifierRegistry(), tmp_path, "synthetic-manifest"
    )
    accepted = run_trajectory(labeler)
    records = [
        json.loads(line)
        for line in (tmp_path / "meta_labels.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3 and len(accepted) == 1
    assert b.state_hash() != initial_hash
    assert all(r["parent_adapter_hash"] == initial_hash for r in records)
    assert all(r["runtime"]["candidate"]["optimizer_steps"] == 3 for r in records)
    assert (tmp_path / "accepted/generation_1/adapter_model.safetensors").exists()


@pytest.mark.integration
def test_qwen_style_non_thinking_prefix_and_completion_masking():
    b = make_tiny_backend()
    # Qwen3 formats the final assistant message with an empty thinking block;
    # generation uses the same block when enable_thinking=False.
    b.tokenizer.chat_template = "{% for m in messages %}{% if m['role']=='assistant' %}{{'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m['content'] + '<|im_end|>\\n'}}{% else %}{{'<|im_start|>user\\n' + m['content'] + '<|im_end|>\\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'}}{% endif %}"
    row = encode_completion(
        b.tokenizer, "one plus two", "#### 3", 128, {"enable_thinking": False}
    )
    assert any(x != -100 for x in row["labels"])
