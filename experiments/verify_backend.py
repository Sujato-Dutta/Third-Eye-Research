"""Offline real LoRA smoke verification on a tiny, randomly initialized Llama.

No model downloads or benchmark performance claims. Use --device cuda inside
an allocated SLURM job; --quantization nf4 also exercises bitsandbytes QLoRA.
"""

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.config import Config, ModelConfig, ProtocolConfig, TrainingConfig
from third_eye.data.schema import Correction, Example
from third_eye.io import write_json


def make_tiny_backend(device="cpu", quantization="none"):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from third_eye.training.hf_backend import HFBackend, seed_everything

    torch.set_num_threads(1)
    seed_everything(42)
    vocab = {
        token: i
        for i, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[EOS]",
                "Question",
                ":",
                "Answer",
                "one",
                "plus",
                "two",
                "three",
                "####",
                "1",
                "2",
                "3",
                "4",
            ]
        )
    }
    core = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=None,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    config = Config(
        model=ModelConfig(
            name="tiny-random-llama-local",
            device=device,
            dtype="bfloat16" if device == "cuda" else "float32",
            quantization=quantization,
            gradient_checkpointing=True,
            chat_kwargs={},
        ),
        training=TrainingConfig(
            rank=2,
            alpha=4,
            max_steps=3,
            learning_rate=0.01,
            gradient_accumulation_steps=2,
            max_sequence_length=64,
        ),
        protocol=ProtocolConfig(candidate_size=2, probe_steps=1, max_new_tokens=8),
    )
    if quantization == "none":
        return HFBackend(config, model=model, tokenizer=tokenizer)
    # NF4 must be applied during loading, not to an already constructed fp32 model.
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        config = replace(config, model=replace(config.model, name=directory))
        backend = HFBackend(config)
    assert torch.cuda.is_available()
    return backend


def verify(device="cpu", quantization="none"):
    import torch

    backend = make_tiny_backend(device, quantization)
    base = {
        name: p.detach().cpu().clone()
        for name, p in backend.model.named_parameters()
        if not p.requires_grad
    }
    example = Example("train-1", "one plus one", "#### 2", "train")
    corrections = [
        Correction(example, "#### 2", 1),
        Correction(Example("train-2", "one plus two", "#### 3", "train"), "#### 3", 1),
    ]
    parent, parent_hash = backend.snapshot(), backend.state_hash()
    loss_before = backend.measure_loss([example])
    report = backend.train(corrections, seed=42)
    updated_hash = backend.state_hash()
    assert updated_hash != parent_hash, "Adapter did not change"
    assert all(
        torch.equal(p.detach().cpu(), base[name])
        for name, p in backend.model.named_parameters()
        if name in base
    ), "Frozen base weights changed"
    assert math.isfinite(report["loss_last"])
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "adapter"
        backend.save_checkpoint(checkpoint, {"purpose": "smoke-verification"})
        backend.restore(parent)
        assert backend.state_hash() == parent_hash, "Rollback failed"
        assert math.isclose(
            backend.measure_loss([example]), loss_before, rel_tol=1e-6
        ), "Restored predictions differ"
        backend.load_checkpoint(checkpoint)
        assert backend.state_hash() == updated_hash, "Checkpoint restore failed"
        # A second update must accumulate from t+1, then restore to that state.
        t1 = backend.snapshot()
        backend.train(corrections, seed=43)
        assert backend.state_hash() != updated_hash
        backend.restore(t1)
        assert backend.state_hash() == updated_hash
        backend.restore(parent)
        backend.train(corrections, seed=42)
        assert backend.state_hash() == updated_hash, (
            "Same parent/seed/update must reproduce the same adapter"
        )
    backend.generate(example.prompt, max_new_tokens=4, seed=42)
    return {
        "status": "passed",
        "device": device,
        "quantization": quantization,
        "model": "tiny random Llama; not a research backbone",
        "adapter_changed": True,
        "base_weights_unchanged": True,
        "rollback_exact": True,
        "checkpoint_reload_exact": True,
        "cumulative_update": True,
        "seed_reproducible": True,
        "generation_exercised": True,
        "training": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--output", default="runs/verification/backend.json")
    args = parser.parse_args()
    report = verify(args.device, args.quantization)
    write_json(args.output, report)
    print(
        f"PASS: {args.device} {args.quantization} training, rollback, checkpoint and reproducibility"
    )


if __name__ == "__main__":
    main()
