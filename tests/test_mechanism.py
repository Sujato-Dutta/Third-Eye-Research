"""Selection/score separation and signed intervention effects on synthetic items."""

import copy

import numpy as np
import pytest

from third_eye.experiments.mechanism import (
    folds,
    matrix,
    state_comparison,
    intervention_export,
)


def fixture():
    parent = {
        "items": {
            r: [
                dict(id=str(i), prompt_sha256=f"p-{r}-{i}", correct=0) for i in range(8)
            ]
            for r in ("target_dev", "ood_dev", "retention_dev")
        }
    }
    assignment = folds(parent)
    m1 = [copy.deepcopy(parent) for _ in range(3)]
    draws = [[copy.deepcopy(parent) for _ in range(3)] for _ in range(3)]
    # M1 prefers candidate0; independent future prefers candidate1.
    for role in parent["items"]:
        for i in range(8):
            m1[0]["items"][role][i]["correct"] = 1
            for draw in draws:
                draw[1]["items"][role][i]["correct"] = 1
    return parent, assignment, m1, draws


def test_future_selection_and_fixed_rule_are_scored_on_heldout_seed_items():
    parent, assignment, m1, draws = fixture()
    result = state_comparison(parent, m1, draws)
    assert result["future_minus_immediate"] == 1
    assert result["immediate_minus_hold"] == 0
    assert result["guarded_update_probability"] == 1
    for role, groups in assignment.items():
        assert not set(groups[0]) & set(groups[1])
        assert sorted(groups[0] + groups[1]) == list(range(8))
    changed = copy.deepcopy(draws)
    # Poison scoring items for candidate0: selector choices must remain fixed,
    # and score must now favor immediate, with a negative signed difference.
    for role in parent["items"]:
        for i in range(8):
            changed[2][0]["items"][role][i]["correct"] = 1
            changed[2][1]["items"][role][i]["correct"] = 0
    assert state_comparison(parent, m1, changed)["future_minus_immediate"] == -1


def test_intervention_primary_holds_the_baseline_candidate_choice():
    parent, _, m1, draws = fixture()
    small_m1 = copy.deepcopy(draws[0])  # prefers candidate1, not baseline0
    small_draws = copy.deepcopy(draws)
    result = state_comparison(parent, m1, draws, small_m1, small_draws)
    assert result["half_lr_same_candidate_minus_baseline"] == 0
    assert result["half_lr_reselected_minus_baseline"] == 1
    zero = [copy.deepcopy(parent) for _ in range(3)]
    assert state_comparison(parent, zero, draws)["guarded_update_probability"] == 0


def test_mismatched_items_and_missing_candidates_fail():
    parent, assignment, m1, draws = fixture()
    m1[0]["items"]["target_dev"][0]["id"] = "wrong"
    with pytest.raises(ValueError, match="items/order"):
        matrix(m1, parent, assignment, 0)
    with pytest.raises(ValueError, match="Three draws"):
        state_comparison(parent, m1, draws[:2])
    assert np.isfinite(
        np.asarray(list(state_comparison(parent, draws[0], draws).values()))
    ).all()


def test_export_cannot_misrepresent_inherited_control_outcomes_as_intervention():
    record = dict(
        continuation={"seed": 12},
        continuation_available=1,
        continuation_adapter_hash="old-control",
        precommit={"features": "baseline-LR"},
        labels={"h1": {"target": 0.2}, "h2": {"target": 0.9}, "utility_h1": 0.9},
        evaluation={"t1": {"target": 0.5}, "t2": {"target": 0.9}},
        evaluation_artifacts={"t1": "new-M1", "t2": "old-M2"},
        runtime={
            "candidate": {"optimizer_steps": 50},
            "continuation": {"optimizer_steps": 50},
        },
        continuation_replicates=[{"seed": 62, "h2": {"target": -0.1}}],
    )
    before = copy.deepcopy(record)
    exported = intervention_export(record)
    assert record == before
    assert exported["labels"] == {"h1": {"target": 0.2}}
    assert exported["continuation_replicates"] == record["continuation_replicates"]
    assert exported["reference_continuation_seed"] == 12
    assert (
        "continuation_adapter_hash" not in exported
        and "continuation_available" not in exported
    )
    assert "precommit" not in exported and "t2" not in exported["evaluation"]
