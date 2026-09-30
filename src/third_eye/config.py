"""Strict, immutable protocol configuration; all candidates share one budget."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from third_eye.io import digest


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen3-4B"
    revision: str = "main"
    quantization: str = "none"
    dtype: str = "bfloat16"
    device: str = "cuda"
    gradient_checkpointing: bool = True
    chat_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})


@dataclass(frozen=True)
class TrainingConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    max_steps: int = 50
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 1024
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class ProtocolConfig:
    seed: int = 42
    candidates: int = 3
    candidate_size: int = 32
    correction_attempts: int = 4
    max_new_tokens: int = 512
    correction_temperature: float = 0.7
    probe_steps: int = 10
    depth: int = 1
    status: str = "pilot"
    # Fractions, not percentage points; frozen before meta-label generation.
    utility_weights: tuple = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)

    def __post_init__(self):
        t, p, m = self.training, self.protocol, self.model
        for value in (
            t.rank,
            t.alpha,
            t.max_steps,
            t.micro_batch_size,
            t.gradient_accumulation_steps,
            p.candidate_size,
            p.correction_attempts,
            p.max_new_tokens,
            p.depth,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Budgets must be positive integers")
        if t.max_sequence_length < 8 or p.candidates != 3:
            raise ValueError("Require sequence length >=8 and K=3")
        if not 0 <= p.probe_steps <= t.max_steps:
            raise ValueError("Probe budget must be within the full update budget")
        if p.depth > 5 or p.status not in {"pilot", "frozen"}:
            raise ValueError("Require T<=5 and status pilot/frozen")
        if m.quantization not in {"none", "nf4"} or m.dtype not in {
            "float32",
            "bfloat16",
        }:
            raise ValueError("Unsupported dtype or quantization")
        if m.quantization == "nf4" and m.device != "cuda":
            raise ValueError("NF4 training requires CUDA")
        if not (math.isfinite(t.learning_rate) and t.learning_rate > 0):
            raise ValueError("Learning rate must be finite and positive")
        if not 0 <= t.dropout < 1 or t.weight_decay < 0 or t.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer/LoRA parameters")
        if (
            not t.target_modules
            or not math.isfinite(p.correction_temperature)
            or p.correction_temperature <= 0
        ):
            raise ValueError("Missing target modules or invalid temperature")
        if (
            len(p.utility_weights) != 3
            or any(not math.isfinite(w) or w < 0 for w in p.utility_weights)
            or not math.isclose(sum(p.utility_weights), 1)
        ):
            raise ValueError("Three nonnegative utility weights must sum to one")

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        if set(raw) != {"model", "training", "protocol"}:
            raise ValueError(
                "Config requires exactly model, training, protocol sections"
            )
        return cls(
            ModelConfig(**raw["model"]),
            TrainingConfig(**raw["training"]),
            ProtocolConfig(**raw["protocol"]),
        )
