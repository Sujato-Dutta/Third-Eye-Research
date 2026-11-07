from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from third_eye.experiments.regime_revision import (
    effective_distance, factor_pairs, lenient_numeric, paired_bootstrap,
    select_regime, unique_batch,
)


@pytest.mark.parametrize("text,expected", [
    ("work\nThe answer is 42 dollars.", Decimal(42)),
    ("work\nThus, 1,200 apples.", Decimal(1200)),
    ("#### 2\nFinal answer: 3", Decimal(3)),
    ("#### 2 dollars", Decimal(2)),
    ("reasoning with 2 and 3", None),
    ("#### 2 or 3", None),
    ("42 is wrong", None),
    ("Therefore -3 kg", Decimal(-3)),
    ("Answer: NaN", None),
])
def test_lenient_parser_is_reference_blind_and_rejects_ambiguous_numbers(text, expected):
    assert lenient_numeric(text) == expected


def test_effective_norm_matches_dense_and_is_gauge_invariant():
    rng = np.random.default_rng(13)
    a0, b0 = rng.normal(size=(2, 7)), rng.normal(size=(5, 2))
    a1, b1 = rng.normal(size=(2, 7)), rng.normal(size=(5, 2))
    left = {"x": {"A": a0, "B": b0}}
    right = {"x": {"A": a1, "B": b1}}
    actual = effective_distance(left, right, 2., 3.)["frobenius_norm"]
    assert actual == pytest.approx(np.linalg.norm(2*b0@a0 - 3*b1@a1))
    gauge = {"x": {"A": a0*4, "B": b0/4}}
    assert effective_distance(left, gauge, 2., 2.)["frobenius_norm"] < 1e-6


def test_effective_increment_is_not_difference_of_factor_norms():
    left = {"x": {"A": np.array([[1., 0.]]), "B": np.array([[1.], [0.]])}}
    right = {"x": {"A": np.array([[0., 1.]]), "B": np.array([[0.], [1.]])}}
    assert effective_distance(left, right, 1., 1.)["frobenius_norm"] == pytest.approx(2**.5)


def test_factor_loader_rejects_missing_and_multiple_adapter_components():
    with pytest.raises(ValueError, match="Complete"):
        factor_pairs({"x.lora_A.weight": np.ones((2, 3))})
    with pytest.raises(ValueError, match="Multiple"):
        factor_pairs({"x.lora_A.a.weight": np.ones((2, 3)),
                      "x.lora_A.b.weight": np.ones((2, 3))})


def test_scarcity_does_not_clone_examples_or_shrink_requested_batch():
    pool = [SimpleNamespace(example=SimpleNamespace(id=str(i))) for i in range(8)]
    assert unique_batch(pool, 32, 42) is None
    assert len(unique_batch(pool, 8, 42)) == 8
    assert unique_batch(pool, 8, 42) == unique_batch(pool[::-1], 8, 42)


def test_positive_regime_requires_all_parents_and_retention_condition():
    rows = [{"model": "qwen3-4b", "recipe": "r0", "seed": seed,
             "available": True, "development_head_gains": [.03, .01, 0.]}
            for seed in range(2026110700, 2026110704)]
    assert select_regime(rows)["recipe"] == "r0"
    assert select_regime(rows[:3]) is None
    rows[-1]["available"] = False
    assert select_regime(rows) is None


def test_paired_bootstrap_preserves_exact_cancellation_on_shared_items():
    states = [{"stream": "qwen3-4b/"+task, "ids": {
        r: [(str(i), "h"+str(i)) for i in range(4)]
        for r in ("target_test", "ood_test", "retention_test")}}
        for task in ("code", "math") for _ in range(2)]
    matrices = [{r: np.zeros((1, 4)) for r in state["ids"]} for state in states]
    value = paired_bootstrap(states, matrices, ["paired"], draws=20)
    assert value["models"]["qwen3-4b"]["paired"] == {
        "mean_pp": 0., "ci95_pp": [0., 0.]}


def test_expanded_development_excludes_training_and_keeps_existing_items():
    from next_round_regimes import fill
    from third_eye.data.schema import Example
    from third_eye.data.benchmarks import canonical_prompt
    old = Example("dev", "existing", "2", "target_dev")
    train_duplicate = Example("different-id", "training problem", "2", "target_dev")
    fresh = Example("fresh", "new held-out problem", "2", "target_dev")
    result = fill([old], [train_duplicate, fresh], 2,
                  {canonical_prompt(train_duplicate)}, "test")
    assert [e.id for e in result] == ["dev", "fresh"]
    assert fill([old], [], 1, set(), "test") == [old]


def test_math_source_ids_match_existing_subject_labels(monkeypatch):
    import next_round_regimes as runner
    monkeypatch.setattr(runner, "MATH_SUBJECTS", ("algebra",))
    monkeypatch.setattr(runner, "source_rows", lambda *args: [
        {"problem": "x+1=2", "solution": r"\boxed{1}", "type": "Algebra"}])
    examples = runner.math_examples("a"*40)
    assert examples[0].id == "math:train:Algebra:0"


@pytest.mark.parametrize("solution,reason", [
    ("No boxed reference", "MATH reference has no boxed answer"),
    (r"\boxed{ }", "MATH reference has an empty boxed answer"),
    (r"\boxed{1", "Unbalanced MATH reference answer"),
])
def test_math_development_records_invalid_source_references_before_sampling(monkeypatch, solution, reason):
    import next_round_regimes as runner
    monkeypatch.setattr(runner, "MATH_SUBJECTS", ("algebra",))
    monkeypatch.setattr(runner, "source_rows", lambda *args: [
        {"problem": "invalid source", "solution": solution, "type": "Algebra"},
        {"problem": "x+1=2", "solution": r"\boxed{1}", "type": "Algebra"}])
    excluded = []
    examples = runner.math_examples("a"*40, excluded)
    assert [e.id for e in examples] == ["math:train:Algebra:1"]
    assert excluded == [{"id": "math:train:Algebra:0", "reason": reason}]
