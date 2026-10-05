"""Validate counterfactual accounting and first-eight-draw preservation."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from third_eye.config import Config
from third_eye.data.schema import Example
from third_eye.updates.corrections import collect_corrections
from retry_prefix_backend import RetryPrefixBackend

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).with_name("audit_retry_budget.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def payload():
    items = [
        {
            "example": {
                "id": str(i),
                "prompt": "problem",
                "answer": "ok",
                "split": "train",
                "difficulty": "default",
            },
            "completion": "ok",
            "attempt": a,
        }
        for i, a in enumerate([1, 8, 9, 16])
    ]
    return {
        "items": items,
        "stats": {
            "correction_attempt_limit": 16,
            "harvest_complete": True,
            "verified_corrections": 4,
            "unresolved_failures": 2,
            "initial_failures": 6,
            "revision_attempts": 66,
        },
    }


def test_counts_include_failed_problems_and_late_successes():
    r, kept = audit.audit_pool(payload(), 8)
    assert r["prefix_revision_attempts"] == 41
    assert r["retained_verified"] == 2
    assert r["saved_revision_attempts"] == 25
    assert [x["attempt"] for x in kept] == [1, 8]


def test_full_cap_has_no_saved_attempts():
    r, _ = audit.audit_pool(payload(), 16)
    assert r["saved_revision_attempts"] == r["lost_verified"] == 0


def test_corrupt_attempt_accounting_rejected():
    p = payload()
    p["stats"]["revision_attempts"] += 1
    with pytest.raises(ValueError, match="accounting"):
        audit.audit_pool(p, 8)


def test_duplicate_ids_rejected():
    p = payload()
    p["items"][1]["example"]["id"] = "0"
    with pytest.raises(ValueError, match="identities"):
        audit.audit_pool(p, 8)


@pytest.mark.parametrize("cap", [0, 17])
def test_invalid_cap_rejected(cap):
    with pytest.raises(ValueError, match="prefix"):
        audit.audit_pool(payload(), cap)


def test_prefix_backend_preserves_per_problem_seeds_and_greedy():
    class Backend:
        def generate(self, prompt, max_new_tokens, seed, temperature=0.0):
            return seed

        def generate_sampled_many(self, prompts, max_new_tokens, seeds, temperature):
            return seeds

    b = RetryPrefixBackend(Backend(), 42)
    assert b.generate("p", 512, 49, 0) == 49
    assert b.generate("p", 512, 100_042 + 3 * 8 + 7, 0.7) == 100_042 + 3 * 16 + 7
    assert b.generate_sampled_many(["p"], 512, [100_042 + 9 * 8 + 1], 0.7) == [
        100_042 + 9 * 16 + 1
    ]


def test_existing_collector_short_prefix_equals_full_pool_prefix():
    class Backend:
        def generate(self, prompt, max_new_tokens, seed, temperature=0.0):
            if temperature == 0:
                return "bad"
            index, attempt = divmod(seed - 100_042, 16)
            return "ok" if attempt + 1 == [2, 9, 16][index] else "bad"

    class Verifier:
        def verify(self, example, completion):
            return completion == "ok"

    examples = [Example(str(i), f"problem {i}", "ok", "train") for i in range(3)]
    p = replace(Config().protocol, correction_attempts=16)
    full, _ = collect_corrections(Backend(), examples, Verifier(), p, 42)
    shorter, stats = collect_corrections(
        RetryPrefixBackend(Backend(), 42),
        examples,
        Verifier(),
        replace(p, correction_attempts=8),
        42,
    )
    assert [c.to_dict() for c in shorter] == [
        c.to_dict() for c in full if c.attempt <= 8
    ]
    assert stats["revision_attempts"] == 18


def test_batched_prefix_preserves_seeds_when_pending_rows_drop_out():
    class Backend:
        def generate_many(self, prompts, max_new_tokens, seeds):
            return ["bad"] * len(prompts)

        def generate_sampled_many(self, prompts, max_new_tokens, seeds, temperature):
            answers = []
            for seed in seeds:
                index, attempt = divmod(seed - 100_042, 16)
                answers.append("ok" if attempt + 1 == [2, 9, 16][index] else "bad")
            return answers

    class Verifier:
        def verify(self, example, completion):
            return completion == "ok"

    examples = [Example(str(i), f"problem {i}", "ok", "train") for i in range(3)]
    p = replace(Config().protocol, correction_attempts=16, sampled_batch_size=16)
    full, _ = collect_corrections(Backend(), examples, Verifier(), p, 42)
    shorter, stats = collect_corrections(
        RetryPrefixBackend(Backend(), 42),
        examples,
        Verifier(),
        replace(p, correction_attempts=8),
        42,
    )
    assert [c.to_dict() for c in shorter] == [
        c.to_dict() for c in full if c.attempt <= 8
    ]
    assert stats["revision_attempts"] == 18
