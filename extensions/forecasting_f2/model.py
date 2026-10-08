"""Compare matched updates and roll out their continuation consequences."""

import torch
from torch import nn

from third_eye.forecasting.models import HistoryEncoder, Direct
from third_eye.forecasting.encoding import FEATURE_NAMES


class ContrastiveContinuation(nn.Module):
    """Permutation-equivariant common drift plus candidate-relative effects.

    The next-state latent is conditioned on *predicted* immediate consequences.
    Actual immediate/future consequences never enter the forward interface.
    """

    def __init__(self, hidden=16, variant="full"):
        super().__init__()
        if variant not in {"full", "no_set", "no_temporal_aux", "no_comparative_loss"}:
            raise ValueError("Unknown F2 ablation")
        self.variant = variant
        n = 2 * len(FEATURE_NAMES)
        self.history = HistoryEncoder(hidden)
        self.context = nn.Sequential(nn.Linear(3 + n + hidden, hidden), nn.Tanh())
        self.immediate_mean = nn.Linear(hidden, 3)
        self.immediate_gate = nn.Linear(hidden, n)
        self.immediate_advantage = nn.Linear(n, 3, bias=False)
        self.next_state = nn.Sequential(nn.Linear(hidden + n + 3, hidden), nn.Tanh())
        self.continuation_mean = nn.Linear(hidden, 3)
        self.continuation_gate = nn.Linear(hidden, n)
        self.continuation_advantage = nn.Linear(n + 3, 3, bias=False)

    def forward(self, state, features, history, mask):
        if features.ndim != 3 or features.shape[1] != 3:
            raise ValueError("F2 requires a complete K=3 candidate set")
        b, k = features.shape[:2]
        h = self.history(history.reshape(b * k, 3, 6), mask.reshape(b * k, 3))
        h = h.reshape(b, k, -1)
        joint = self.variant != "no_set"
        center = features - features.mean(1, keepdim=True) if joint else features
        context_features = (
            features.mean(1, keepdim=True).expand_as(features) if joint else features
        )
        context = self.context(torch.cat((state, context_features, h), dim=-1))
        advantage1 = self.immediate_advantage(
            center * torch.sigmoid(self.immediate_gate(context))
        )
        if joint:
            advantage1 = advantage1 - advantage1.mean(1, keepdim=True)
        immediate = self.immediate_mean(context) + advantage1
        # Normalized capability deltas are restored to their physical units by
        # the adapter; the rollout latent uses this learned representation only.
        z1 = self.next_state(torch.cat((context, center, immediate), dim=-1))
        relative1 = immediate - immediate.mean(1, keepdim=True) if joint else immediate
        advantage_c = self.continuation_advantage(
            torch.cat(
                (center * torch.sigmoid(self.continuation_gate(z1)), relative1), dim=-1
            )
        )
        pooled_z1 = z1.mean(1, keepdim=True).expand_as(z1) if joint else z1
        if joint:
            advantage_c = advantage_c - advantage_c.mean(1, keepdim=True)
        continuation = self.continuation_mean(pooled_z1) + advantage_c
        return {
            "prediction": immediate + continuation,
            "h1": immediate,
            "continuation": continuation,
            "advantage_h1": advantage1,
            "advantage_continuation": advantage_c,
        }


class IndependentControl(nn.Module):
    """Original GRU architecture with the same comparative fitting objective."""

    def __init__(self):
        super().__init__()
        self.original = Direct(hidden=64)

    def forward(self, state, features, history, mask):
        b, k = features.shape[:2]
        result = self.original(
            state.reshape(b * k, 3),
            features.reshape(b * k, -1),
            history.reshape(b * k, 3, 6),
            mask.reshape(b * k, 3),
        )
        return {"prediction": result["prediction"].reshape(b, k, 3)}


def build_model(variant="full", hidden=16):
    model = (
        IndependentControl()
        if variant == "independent"
        else ContrastiveContinuation(hidden, variant)
    )
    if sum(p.numel() for p in model.parameters()) >= 1_000_000:
        raise ValueError("F2 exceeds the frozen forecaster parameter ceiling")
    return model
