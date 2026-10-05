"""Single-device LoRA/QLoRA with cumulative adapters and exact branch rollback."""

from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from third_eye.io import append_jsonl, file_digest, write_json
from third_eye.training.tokenization import collate, encode_completion, format_prompt
from third_eye.training.sources import resolve_source


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HFBackend:
    def __init__(self, config, model=None, tokenizer=None):
        self.config = config
        if config.protocol.deterministic_execution:
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {None, ":4096:8"}:
                raise RuntimeError("A2 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        m, t = config.model, config.training
        source_name, source_revision, self.source_proof = resolve_source(m)
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
            source_name, revision=source_revision, use_fast=True
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
                "revision": source_revision,
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
            base_config = AutoConfig.from_pretrained(
                source_name, revision=source_revision
            )
            if base_config.model_type == "gemma3":
                # 4B Gemma has a multimodal outer config; load its language
                # module with the proper composite class, without vision LoRA.
                from transformers import Gemma3ForConditionalGeneration

                model = Gemma3ForConditionalGeneration.from_pretrained(
                    source_name, **kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(source_name, **kwargs)
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
                target_modules=(
                    r".*language_model.*\.(?:" + "|".join(t.target_modules) + r")$"
                    if getattr(model.config, "model_type", None) == "gemma3"
                    else list(t.target_modules)
                ),
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
            self.source_proof["original_revision"]
            if self.source_proof
            else getattr(model.config, "_commit_hash", None) or m.revision
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
        text_config = (
            self.model.config.get_text_config()
            if hasattr(self.model.config, "get_text_config")
            else self.model.config
        )
        capacity = getattr(text_config, "max_position_embeddings", None)
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

    def generate_many(self, prompts, max_new_tokens, seeds=None):
        """Greedy decoding in fixed left-padded batches; sampled retries stay seeded."""
        prompts = list(prompts)
        seeds = list(seeds) if seeds is not None else [0] * len(prompts)
        if len(seeds) != len(prompts):
            raise ValueError("Generation seeds must match prompts")
        batch_size = self.config.protocol.generation_batch_size
        if batch_size == 1:
            return [
                self.generate(p, max_new_tokens, seed=s) for p, s in zip(prompts, seeds)
            ]
        self.model.eval()
        results = []
        for start in range(0, len(prompts), batch_size):
            seed_everything(seeds[start])
            texts = [
                format_prompt(self.tokenizer, p, self.config.model.chat_kwargs)
                for p in prompts[start : start + batch_size]
            ]
            padding = self.tokenizer.padding_side
            try:
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_token_type_ids=False,
                ).to(self.device)
            finally:
                self.tokenizer.padding_side = padding
            text_config = (
                self.model.config.get_text_config()
                if hasattr(self.model.config, "get_text_config")
                else self.model.config
            )
            capacity = getattr(text_config, "max_position_embeddings", None)
            length = inputs.input_ids.shape[1]
            if capacity and length + max_new_tokens > capacity:
                raise ValueError("Generation exceeds model context")
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            results.extend(
                self.tokenizer.batch_decode(
                    output[:, length:], skip_special_tokens=True
                )
            )
        return results

    def predict_choices(self, prompts):
        """Rank A/B/C/D by conditional log likelihood, excluding end tokens.

        Teacher forcing scores the answer suffix only, including a tokenizer
        boundary token when it crosses into that suffix. It never generates
        explanations or conditions on the reference answer.
        """
        self.model.eval()
        result = []
        batch_size = self.config.protocol.generation_batch_size
        for start in range(0, len(prompts), batch_size):
            prefixes = [
                format_prompt(self.tokenizer, p, self.config.model.chat_kwargs)
                for p in prompts[start : start + batch_size]
            ]
            texts = [prefix + choice for prefix in prefixes for choice in "ABCD"]
            padding = self.tokenizer.padding_side
            try:
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                    return_token_type_ids=False,
                )
            finally:
                self.tokenizer.padding_side = padding
            offsets = inputs.pop("offset_mapping")
            lengths = torch.tensor([len(prefix) for prefix in prefixes for _ in "ABCD"])
            answer_mask = offsets[:, :, 1] > lengths[:, None]
            counts = answer_mask.sum(1).tolist()
            if not counts or min(counts) < 1:
                raise ValueError("Multiple-choice hypotheses have no answer tokens")
            text_config = (
                self.model.config.get_text_config()
                if hasattr(self.model.config, "get_text_config")
                else self.model.config
            )
            capacity = getattr(text_config, "max_position_embeddings", None)
            if capacity and inputs.input_ids.shape[1] > capacity:
                raise ValueError("Choice scoring exceeds model context")
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                logits = self.model(
                    **inputs, use_cache=False, logits_to_keep=max(counts) + 1
                ).logits.float()
                logp = torch.log_softmax(logits, dim=-1)
                scores = []
                for index, count in enumerate(counts):
                    chosen = inputs.input_ids[index, -count:]
                    scores.append(
                        logp[index, -count - 1 : -1].gather(-1, chosen[:, None]).sum()
                    )
                winners = torch.stack(scores).reshape(-1, 4).argmax(1).tolist()
            result.extend("ABCD"[winner] for winner in winners)
        return result

    def generate_sampled_many(self, prompts, max_new_tokens, seeds, temperature):
        from third_eye.training.sampling import SeededSampling

        prompts, seeds = list(prompts), list(seeds)
        if len(prompts) != len(seeds):
            raise ValueError("Sampling seeds must match prompts")
        size = self.config.protocol.sampled_batch_size
        if size == 1:
            return [
                self.generate(p, max_new_tokens, seed=s, temperature=temperature)
                for p, s in zip(prompts, seeds)
            ]
        self.model.eval()
        results = []
        for start in range(0, len(prompts), size):
            texts = [
                format_prompt(self.tokenizer, p, self.config.model.chat_kwargs)
                for p in prompts[start : start + size]
            ]
            padding = self.tokenizer.padding_side
            try:
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_token_type_ids=False,
                ).to(self.device)
            finally:
                self.tokenizer.padding_side = padding
            text_config = (
                self.model.config.get_text_config()
                if hasattr(self.model.config, "get_text_config")
                else self.model.config
            )
            capacity = getattr(text_config, "max_position_embeddings", None)
            length = inputs.input_ids.shape[1]
            if capacity and length + max_new_tokens > capacity:
                raise ValueError("Sampling exceeds model context")
            sampler = SeededSampling(
                seeds[start : start + size],
                self.device,
                temperature,
                self.model.generation_config,
            )
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    logits_processor=[sampler],
                )
            results.extend(
                self.tokenizer.batch_decode(
                    output[:, length:], skip_special_tokens=True
                )
            )
        return results

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

    def _gradient(self, examples):
        """Adapter gradients at the current parent; no optimizer mutation."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        try:
            for example in examples:
                row = self._encode(example.prompt, example.answer)
                batch = {
                    k: v.to(self.device)
                    for k, v in collate([row], self.tokenizer.pad_token_id).items()
                }
                with self._autocast():
                    loss = self.model(**batch).loss / len(examples)
                loss.backward()
            parts = [
                p.grad.detach().float().cpu().flatten()
                if p.grad is not None
                else torch.zeros(p.numel())
                for p in self.model.parameters()
                if p.requires_grad
            ]
            return torch.cat(parts), [float(part.norm()) for part in parts]
        finally:
            self.model.zero_grad(set_to_none=True)

    def candidate_diagnostics(self, examples, anchor):
        lengths = [
            len(self.tokenizer.encode(e.answer, add_special_tokens=False))
            for e in examples
        ]
        losses = [self.measure_loss([e]) for e in examples]
        candidate, norms = self._gradient(examples[:8])
        retention, _ = self._gradient(anchor)
        denom = candidate.norm() * retention.norm()
        valid = (candidate != 0) & (retention != 0)
        embeddings = []
        with torch.inference_mode():
            for ex in examples:
                ids = torch.tensor(
                    self.tokenizer.encode(ex.prompt, add_special_tokens=False),
                    device=self.device,
                )
                vector = self.model.get_input_embeddings()(ids).float().mean(dim=0)
                embeddings.append(torch.nn.functional.normalize(vector, dim=0).cpu())
        matrix = torch.stack(embeddings)
        similarity = matrix @ matrix.T
        diversity = (
            (
                1
                - float(
                    (similarity.sum() - similarity.trace())
                    / (len(examples) * (len(examples) - 1))
                )
            )
            if len(examples) > 1
            else 0.0
        )
        return {
            "completion_tokens_mean": float(np.mean(lengths)),
            "completion_tokens_std": float(np.std(lengths)),
            "confidence_mean": float(np.mean(np.exp(-np.asarray(losses)))),
            "gradient_norm": float(candidate.norm()),
            "retention_gradient_cosine": float(torch.dot(candidate, retention) / denom)
            if denom > 0
            else 0.0,
            "gradient_sign_agreement": float(
                (candidate[valid].sign() == retention[valid].sign()).float().mean()
            )
            if valid.any()
            else 0.0,
            "layer_gradient_norm_mean": float(np.mean(norms)),
            "layer_gradient_norm_std": float(np.std(norms)),
            "embedding_diversity": diversity,
        }

    def anchor_distribution(self, anchor):
        self.model.eval()
        distributions = []
        with torch.inference_mode():
            for example in anchor:
                text = format_prompt(
                    self.tokenizer, example.prompt, self.config.model.chat_kwargs
                )
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_token_type_ids=False,
                ).to(self.device)
                with self._autocast():
                    logits = self.model(**inputs).logits[0, -1].float()
                distributions.append(logits.log_softmax(dim=-1).cpu())
        return torch.stack(distributions)

    def probe_diagnostics(self, parent, anchor, distribution):
        after = self.anchor_distribution(anchor)
        kl = (distribution.exp() * (distribution - after)).sum(dim=-1).mean()
        squared = sum(
            float((v.float() - parent[k].float()).square().sum())
            for k, v in self.snapshot().items()
        )
        return {
            "probe_anchor_kl": max(0.0, float(kl)),
            "probe_adapter_delta_norm": squared**0.5,
        }

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
                "model_source": self.source_proof,
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
            or saved.get("model_source") != self.source_proof
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
