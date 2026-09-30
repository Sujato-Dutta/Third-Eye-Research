from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Consequences:
    target: float
    ood: float
    retention: float

    def to_dict(self):
        return asdict(self)

    def delta(self, baseline):
        return Consequences(
            self.target - baseline.target,
            self.ood - baseline.ood,
            self.retention - baseline.retention,
        )

    def utility(self, weights):
        return sum(
            w * v for w, v in zip(weights, (self.target, self.ood, self.retention))
        )


def evaluate(backend, splits, verifier, max_new_tokens, seed):
    metrics = []
    for role in ("target_dev", "ood_dev", "retention_dev"):
        examples = splits[role]
        passed = sum(
            verifier.verify(ex, backend.generate(ex.prompt, max_new_tokens, seed=seed))
            for ex in examples
        )
        metrics.append(passed / len(examples))
    return Consequences(*metrics)
