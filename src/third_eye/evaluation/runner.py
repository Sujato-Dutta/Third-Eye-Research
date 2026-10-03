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


def generate_greedy(backend, examples, max_new_tokens, seeds):
    if len(seeds) != len(examples):
        raise ValueError("Generation seeds must match examples")
    protocol = getattr(getattr(backend, "config", None), "protocol", None)
    if getattr(
        protocol, "multiple_choice_scoring", "generation"
    ) == "conditional_likelihood" and all(
        ex.metadata.get("verifier") == "multiple_choice" for ex in examples
    ):
        completions = backend.predict_choices([ex.prompt for ex in examples])
    elif hasattr(backend, "generate_many"):
        completions = backend.generate_many(
            [ex.prompt for ex in examples], max_new_tokens, seeds
        )
    else:
        completions = [
            backend.generate(ex.prompt, max_new_tokens, seed=seed)
            for ex, seed in zip(examples, seeds)
        ]
    if len(completions) != len(examples):
        raise ValueError("Generation returned an incomplete batch")
    return completions


def verify_completions(verifier, examples, completions):
    if len(examples) != len(completions):
        raise ValueError("Verifier inputs must match")
    if hasattr(verifier, "verify_many"):
        results = verifier.verify_many(examples, completions)
    else:
        results = [verifier.verify(ex, text) for ex, text in zip(examples, completions)]
    if len(results) != len(examples):
        raise ValueError("Verifier returned an incomplete batch")
    return results


def evaluate(backend, splits, verifier, max_new_tokens, seed):
    metrics = []
    for role in ("target_dev", "ood_dev", "retention_dev"):
        examples = splits[role]
        print(f"Evaluating {role}: {len(examples)} examples", flush=True)
        completions = generate_greedy(
            backend, examples, max_new_tokens, [seed] * len(examples)
        )
        passed = sum(verify_completions(verifier, examples, completions))
        metrics.append(passed / len(examples))
        print(f"Completed {role}: {passed}/{len(examples)}", flush=True)
    return Consequences(*metrics)
