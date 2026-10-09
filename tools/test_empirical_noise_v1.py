"""Synthetic checks: reject nonidentical replay and preserve aligned item rows."""

from types import SimpleNamespace
import pytest
import empirical_noise_v1 as noise
from third_eye.data.schema import Example


class Backend:
    config = SimpleNamespace(
        protocol=SimpleNamespace(multiple_choice_scoring="generation")
    )

    def state_hash(self):
        return "adapter"

    def generate_many(self, prompts, max_new_tokens, seeds):
        return ["correct" if p == "p0" else "wrong" for p in prompts]


class Verifier:
    def verify_many(self, examples, texts):
        return [t == "correct" for t in texts]


def test_aligned_collector_includes_every_role_and_exact_counts():
    splits = {
        r: [Example(str(i), f"p{i}", "answer", r) for i in range(2)]
        for r in noise.ROLES
    }
    measured = noise.collect_items(
        Backend(), splits, Verifier(), SimpleNamespace(max_new_tokens=512), 42
    )
    noise.exact_metrics(measured, dict(target=0.5, ood=0.5, retention=0.5), "synthetic")
    assert measured["adapter_hash"] == "adapter"
    assert all(
        [x["correct"] for x in measured["items"][r]] == [1, 0] for r in noise.ROLES
    )
    assert all(
        [x["id"] for x in measured["items"][r]] == ["0", "1"] for r in noise.ROLES
    )
    with pytest.raises(RuntimeError):
        noise.exact_metrics(measured, dict(target=1.0, ood=0.5, retention=0.5), "bad")


def test_exact_replay_mismatch_never_silently_accepts():
    assert noise.same_hash(Backend(), "adapter", "synthetic") == "adapter"
    with pytest.raises(RuntimeError, match="Exact replay failed"):
        noise.same_hash(Backend(), "different", "synthetic")


def test_duplicate_item_ids_stop_measurement():
    splits = {r: [Example("same", "p0", "answer", r)] * 2 for r in noise.ROLES}
    with pytest.raises(RuntimeError, match="unique"):
        noise.collect_items(
            Backend(), splits, Verifier(), SimpleNamespace(max_new_tokens=512), 42
        )
