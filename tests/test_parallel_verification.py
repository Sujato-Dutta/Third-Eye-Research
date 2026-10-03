from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest

from third_eye.config import Config
from third_eye.data.schema import Example
from third_eye.evaluation.benchmarks import BenchmarkVerifier
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.updates.corrections import collect_corrections


def test_round_based_corrections_preserve_seeds_pool_order_and_counts():
    class Backend:
        def __init__(self):
            self.prompts, self.batches = [], []

        def generate(self, prompt, max_new_tokens, seed=0, **kwargs):
            self.prompts.append(prompt)
            return {100042: "#### 10", 100047: "#### 20", 45: "#### 40"}.get(
                seed, "#### -1"
            )

        def generate_sampled_many(self, prompts, max_new_tokens, seeds, temperature):
            self.batches.append(seeds)
            return [
                self.generate(p, max_new_tokens, seed=s, temperature=temperature)
                for p, s in zip(prompts, seeds)
            ]

    examples = [
        Example(
            str(i), "opaque problem " + "ABCD"[i], "#### " + str(10 * (i + 1)), "train"
        )
        for i in range(4)
    ]
    protocol = Config().protocol
    serial, parallel = Backend(), Backend()
    expected, old_stats = collect_corrections(
        serial, examples, VerifierRegistry(), protocol, 42
    )
    actual, new_stats = collect_corrections(
        parallel,
        examples,
        VerifierRegistry(),
        replace(protocol, sampled_batch_size=3),
        42,
    )
    assert [c.to_dict() for c in actual] == [c.to_dict() for c in expected]
    for name in (
        "initial_failures",
        "revision_attempts",
        "verified_corrections",
        "generated_completions",
        "generated_words",
        "verifier_pass_rate",
    ):
        assert old_stats[name] == new_stats[name]
    assert parallel.batches[:2] == [[100042, 100046, 100050], [100047, 100051]]
    assert all(
        not any(answer in p for answer in ["10", "20", "30"]) for p in parallel.prompts
    )


def test_parallel_verifier_preserves_order_and_propagates_failure(monkeypatch):
    monkeypatch.setenv("THIRD_EYE_VERIFIER_WORKERS", "3")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    barrier = threading.Barrier(3)
    threads = set()

    def verify(ex, text):
        threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        return text == ex.answer

    verifier = BenchmarkVerifier(SimpleNamespace(verify=verify))
    examples = [
        Example(str(i), "code prompt", "good", "train", task="code") for i in range(3)
    ]
    assert verifier.verify_many(examples, ["good", "bad", "good"]) == [
        True,
        False,
        True,
    ]
    assert len(threads) == 3

    def fail(ex, text):
        raise RuntimeError("Sandbox unavailable")

    verifier.sandbox = SimpleNamespace(verify=fail)
    with pytest.raises(RuntimeError, match="Sandbox unavailable"):
        VerifierRegistry(external=verifier).verify_many(examples, ["good"] * 3)
    monkeypatch.setenv("THIRD_EYE_VERIFIER_WORKERS", "4")
    with pytest.raises(ValueError, match="allocated CPUs"):
        verifier.verify_many(examples, ["good"] * 3)


@pytest.mark.integration
def test_parallel_symbolic_verification_agrees_with_serial(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    examples = [
        Example(
            str(i),
            "math prompt",
            answer,
            "ood_dev",
            metadata={"verifier": "symbolic_math"},
        )
        for i, answer in enumerate([r"\frac{1}{2}", "2", "3"])
    ]
    completions = [r"$\boxed{0.5}$", r"$\boxed{4}$", r"$\boxed{3}$"]
    verifier = BenchmarkVerifier()
    monkeypatch.setenv("THIRD_EYE_VERIFIER_WORKERS", "1")
    expected = verifier.verify_many(examples, completions)
    monkeypatch.setenv("THIRD_EYE_VERIFIER_WORKERS", "2")
    monkeypatch.setenv("THIRD_EYE_SYMBOLIC_WORKERS", "2")
    assert expected == [True, False, True]
    assert verifier.verify_many(examples, completions) == expected
