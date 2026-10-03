"""Independent per-example random streams for batched correction generation."""

import torch
from transformers import (
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class SeededSampling(LogitsProcessor):
    """Sample independently, then let greedy generation handle EOS/padding.

    Applies the same temperature, top-k and top-p transformations as the
    supported single-example generation path. Each row owns a generator;
    another row finishing early cannot change its random stream.
    """

    def __init__(self, seeds, device, temperature, generation_config):
        if generation_config.num_beams != 1:
            raise ValueError("Batched corrections require single-beam sampling")
        for name, neutral in (
            ("min_p", 0),
            ("typical_p", 1),
            ("epsilon_cutoff", 0),
            ("eta_cutoff", 0),
        ):
            if getattr(generation_config, name, None) not in (None, neutral):
                raise ValueError("Unsupported correction sampling parameter: " + name)
        self.generators = [
            torch.Generator(device=device).manual_seed(seed) for seed in seeds
        ]
        self.warpers = [TemperatureLogitsWarper(temperature)]
        if generation_config.top_k:
            self.warpers.append(TopKLogitsWarper(generation_config.top_k))
        self.warpers.append(TopPLogitsWarper(0.95))

    def __call__(self, input_ids, scores):
        if len(self.generators) != len(scores):
            raise ValueError("Sampling seeds must match batch rows")
        for warper in self.warpers:
            scores = warper(input_ids, scores)
        probabilities = scores.softmax(-1)
        tokens = torch.stack(
            [
                torch.multinomial(row, 1, generator=generator)
                for row, generator in zip(probabilities, self.generators)
            ]
        )
        return torch.full_like(scores, -torch.inf).scatter_(1, tokens, 0)
