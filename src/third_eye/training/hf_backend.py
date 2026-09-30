"""Single-device LoRA/QLoRA with cumulative adapters and exact branch rollback."""

from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from peft.utils.save_and_load import load_peft_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from third_eye.io import append_jsonl, file_digest, write_json
from third_eye.training.tokenization import collate, encode_completion, format_prompt


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HFBackend:
    def __init__(self, config, model=None, tokenizer=None):
        self.config = config
        m, t = config.model, config.training
        self.device = torch.device(m.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; allocate a GPU or use a CPU smoke config"
            )
        if (
            m.dtype == "bfloat16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Configured GPU does not support bf16")
        seed_everything(config.protocol.seed)
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            m.name, revision=m.revision, use_fast=True
        )
        if not tokenizer.is_fast:
            raise ValueError(
                "A fast tokenizer is required for precise assistant-token masking"
            )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer needs an existing EOS/pad token; don't mutate frozen vocabulary"
                )
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        dtype = getattr(torch, m.dtype)
        if model is None:
            kwargs = {
                "revision": m.revision,
                "torch_dtype": dtype,
                "trust_remote_code": False,
            }
            if m.quantization == "nf4":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
                kwargs["device_map"] = {"": torch.cuda.current_device()}
            model = AutoModelForCausalLM.from_pretrained(m.name, **kwargs)
        if m.quantization == "nf4":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=m.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            model.to(device=self.device, dtype=dtype)
        self.model = get_peft_model(
            model,
            LoraConfig(
                r=t.rank,
                lora_alpha=t.alpha,
                lora_dropout=t.dropout,
                target_modules=list(t.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        if m.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        if hasattr(self.model.config, "get_text_config"):
            self.model.config.get_text_config().use_cache = False
        self.model.eval()
        self.resolved_revision = (
            getattr(model.config, "_commit_hash", None) or m.revision
        )

    def snapshot(self):
        # Only the adapter changes. Copies on CPU avoid K simultaneous GPU models.
        return {
            name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(
                self.model, save_embedding_layers=False
            ).items()
        }

    def restore(self, snapshot):
        expected = self.snapshot()
        if set(expected) != set(snapshot) or any(
            expected[k].shape != snapshot[k].shape for k in expected
        ):
            raise ValueError("Adapter snapshot is incompatible")
        set_peft_model_state_dict(self.model, snapshot)
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    def state_hash(self):
        result = hashlib.sha256()
        for name, tensor in sorted(self.snapshot().items()):
            result.update(name.encode())
            result.update(str((tensor.shape, tensor.dtype)).encode())
            result.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        return result.hexdigest()

    def generate(self, prompt, max_new_tokens, seed=0, temperature=0.0):
        seed_everything(seed)
        self.model.eval()
        text = format_prompt(self.tokenizer, prompt, self.config.model.chat_kwargs)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
        ).to(self.device)
        capacity = getattr(self.model.config, "max_position_embeddings", None)
        if capacity and inputs.input_ids.shape[1] + max_new_tokens > capacity:
            raise ValueError(
                "Generation exceeds model context; shorten prompt or token budget"
            )
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=0.95)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

    def _encode(self, prompt, completion):
        return encode_completion(
            self.tokenizer,
            prompt,
            completion,
            self.config.training.max_sequence_length,
            self.config.model.chat_kwargs,
        )

    def _autocast(self):
        return (
            torch.autocast(self.device.type, dtype=torch.bfloat16)
            if self.config.model.dtype == "bfloat16"
            else nullcontext()
        )

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def train(self, corrections, seed, max_steps=None, log_path=None):
        if not corrections:
            raise ValueError("Cannot train an empty update")
        if any(item.example.split != "train" for item in corrections):
            raise ValueError("Non-training examples rejected by backend")
        t = self.config.training
        steps = t.max_steps if max_steps is None else max_steps
        if steps < 1 or steps > t.max_steps:
            raise ValueError("Invalid update/probe budget")
        rows = [
            self._encode(item.example.prompt, item.completion) for item in corrections
        ]
        seed_everything(seed)
        generator = random.Random(seed)
        order, cursor = list(range(len(rows))), 0
        generator.shuffle(order)
        params = [p for p in self.model.parameters() if p.requires_grad]
        # Fresh optimizer per update by design; momentum must not cross branches.
        optimizer = torch.optim.AdamW(
            params, lr=t.learning_rate, weight_decay=t.weight_decay
        )
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._sync()
        start, logs = time.perf_counter(), []
        try:
            for step in range(steps):
                loss_sum = 0.0
                for _ in range(t.gradient_accumulation_steps):
                    chosen = []
                    for _ in range(t.micro_batch_size):
                        if cursor == len(order):
                            generator.shuffle(order)
                            cursor = 0
                        chosen.append(rows[order[cursor]])
                        cursor += 1
                    batch = {
                        k: v.to(self.device)
                        for k, v in collate(chosen, self.tokenizer.pad_token_id).items()
                    }
                    with self._autocast():
                        loss = self.model(**batch).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "Non-finite loss; update will be rolled back by orchestration"
                        )
                    (loss / t.gradient_accumulation_steps).backward()
                    loss_sum += loss.detach().float().item()
                layer_norms = {
                    name: float(p.grad.detach().float().norm().item())
                    for name, p in self.model.named_parameters()
                    if p.requires_grad and p.grad is not None
                }
                norm = torch.nn.utils.clip_grad_norm_(params, t.max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                record = {
                    "step": step + 1,
                    "loss": loss_sum / t.gradient_accumulation_steps,
                    "gradient_norm": float(norm),
                    "layerwise_adapter_gradient_norms": layer_norms,
                }
                logs.append(record)
                if log_path:
                    append_jsonl(log_path, record)
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.eval()
            del optimizer
        self._sync()
        return {
            "optimizer_steps": steps,
            "examples": len(rows),
            "seed": seed,
            "seconds": time.perf_counter() - start,
            "loss_first": logs[0]["loss"],
            "loss_last": logs[-1]["loss"],
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()
            if self.device.type == "cuda"
            else 0,
            "trainable_parameters": sum(p.numel() for p in params),
        }

    def measure_loss(self, examples):
        if not examples:
            raise ValueError("Loss measurement needs examples")
        self.model.eval()
        total, tokens = 0.0, 0
        with torch.inference_mode():
            for example in examples:
                row = self._encode(example.prompt, example.answer)
                batch = {
                    k: v.to(self.device)
                    for k, v in collate([row], self.tokenizer.pad_token_id).items()
                }
                count = int((batch["labels"][:, 1:] != -100).sum())
                with self._autocast():
                    loss = self.model(**batch).loss
                total += loss.item() * count
                tokens += count
        return total / tokens

    def save_checkpoint(self, path, metadata):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(
            path, safe_serialization=True, save_embedding_layers=False
        )
        self.tokenizer.save_pretrained(path)
        weights = {p.name: file_digest(p) for p in path.glob("*.safetensors")}
        write_json(
            path / "state.json",
            {
                "schema_version": 1,
                "config": self.config.to_dict(),
                "config_hash": self.config.fingerprint,
                "adapter_hash": self.state_hash(),
                "resolved_revision": self.resolved_revision,
                "weights": weights,
                "versions": {
                    pkg: importlib.metadata.version(pkg)
                    for pkg in ("torch", "transformers", "peft", "accelerate")
                },
                "metadata": metadata,
            },
        )

    def load_checkpoint(self, path):
        path = Path(path)
        saved = json.loads((path / "state.json").read_text())
        if (
            saved["config_hash"] != self.config.fingerprint
            or saved["resolved_revision"] != self.resolved_revision
        ):
            raise ValueError(
                "Checkpoint protocol/backbone differs; use its exact config and model revision"
            )
        for filename, sha in saved["weights"].items():
            if file_digest(path / filename) != sha:
                raise ValueError("Checkpoint checksum mismatch")
        self.restore(load_peft_weights(path, device="cpu"))
        if self.state_hash() != saved["adapter_hash"]:
            raise ValueError("Restored adapter hash differs")
        return saved
