"""Diagnostic adapter that shortens the cap without changing A1 random draws."""


class RetryPrefixBackend:
    def __init__(self, backend, harvest_seed, cap=8, seed_stride=16):
        if not 1 <= cap <= seed_stride:
            raise ValueError("Require a positive retry prefix")
        self.backend, self.harvest_seed = backend, harvest_seed
        self.cap, self.seed_stride = cap, seed_stride

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def remap(self, seed):
        delta = seed - self.harvest_seed - 100_000
        if delta < 0:
            raise ValueError("Unexpected correction seed")
        index, attempt = divmod(delta, self.cap)
        return self.harvest_seed + 100_000 + index * self.seed_stride + attempt

    def generate_sampled_many(self, prompts, max_new_tokens, seeds, temperature):
        return self.backend.generate_sampled_many(
            prompts, max_new_tokens, [self.remap(s) for s in seeds], temperature
        )

    def generate(self, prompt, max_new_tokens, seed, temperature=0.0):
        # Greedy initial answers must retain the unmodified initial-answer seeds.
        corrected_seed = self.remap(seed) if temperature > 0 else seed
        return self.backend.generate(
            prompt, max_new_tokens, corrected_seed, temperature
        )
