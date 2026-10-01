"""Compact H=1 MLP, temporal multi-head Direct, and latent Dynamics."""

import torch
from torch import nn

from third_eye.forecasting.encoding import FEATURE_NAMES


class HistoryEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(6, hidden, num_layers=2, batch_first=True)

    def forward(self, history, mask):
        hidden = history.new_zeros(2, history.shape[0], self.gru.hidden_size)
        for i in range(history.shape[1]):
            _, updated = self.gru(history[:, i : i + 1], hidden)
            hidden = torch.where(mask[:, i][None, :, None].bool(), updated, hidden)
        return hidden[-1]


class OneStep(nn.Module):
    def __init__(self, hidden=64, scalar=False):
        super().__init__()
        self.scalar = scalar
        self.net = nn.Sequential(
            nn.Linear(3 + 2 * len(FEATURE_NAMES) + 18 + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1 if scalar else 3),
        )

    def forward(self, state, features, history, mask):
        result = self.net(
            torch.cat((state, features, history.flatten(1), mask), dim=-1)
        )
        return {"prediction": result}


class Direct(nn.Module):
    def __init__(self, hidden=64, scalar=False):
        super().__init__()
        self.scalar = scalar
        self.history = HistoryEncoder(hidden)
        self.net = nn.Sequential(
            nn.Linear(3 + 2 * len(FEATURE_NAMES) + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1 if scalar else 3),
        )

    def forward(self, state, features, history, mask):
        return {
            "prediction": self.net(
                torch.cat((state, features, self.history(history, mask)), dim=-1)
            )
        }


class Dynamics(nn.Module):
    """Roll out a shared residual transition, predicting the continuation.

    The actual continuation batch is unavailable before commitment. A learned
    standardized-continuation embedding is conditioned on predicted z1.
    H=1/H=2 labels supervise decoding; latent consistency uses post-update
    capability encodings only as training targets (never inference inputs).
    """

    def __init__(self, hidden=64, scalar=False):
        super().__init__()
        if scalar:
            raise ValueError("Dynamics requires multi-consequence supervision")
        self.history = HistoryEncoder(hidden)
        self.state_encoder = nn.Sequential(nn.Linear(3 + hidden, hidden), nn.Tanh())
        self.update_encoder = nn.Sequential(
            nn.Linear(2 * len(FEATURE_NAMES), hidden), nn.Tanh()
        )
        self.transition = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )
        self.continuation = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )
        self.decoder = nn.Linear(hidden, 3)

    def encode_state(self, state, context):
        return self.state_encoder(torch.cat((state, context), dim=-1))

    def forward(self, state, features, history, mask):
        context = self.history(history, mask)
        z0 = self.encode_state(state, context)
        u = self.update_encoder(features)
        z1 = z0 + self.transition(torch.cat((z0, u), dim=-1))
        z2 = z1 + self.transition(torch.cat((z1, self.continuation(z1)), dim=-1))
        base = self.decoder(z0)
        return {
            "prediction": self.decoder(z2) - base,
            "h1": self.decoder(z1) - base,
            "z1": z1,
            "z2": z2,
            "context": context,
        }


def build_model(kind, hidden=64, scalar=False):
    models = {
        "one_step": OneStep,
        "direct": Direct,
        "dynamics": Dynamics,
        "matched_h1": Direct,
    }
    if kind not in models:
        raise ValueError("Unknown forecaster architecture")
    model = models[kind](hidden, scalar)
    if sum(p.numel() for p in model.parameters()) >= 1_000_000:
        raise ValueError("Forecasters must have fewer than one million parameters")
    return model
