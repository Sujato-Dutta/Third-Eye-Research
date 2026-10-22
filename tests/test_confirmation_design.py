"""Design gates must price all work and distinguish measurement from efficacy."""

import copy

import numpy as np
import pytest

from test_admission_audit import data
from third_eye.experiments.admission_audit import evaluate
from third_eye.experiments.confirmation_design import (
    bootstrap,
    choose_design,
    conditional_power,
    policy_actions,
    price,
)
from third_eye.experiments.mechanism import ROLES, folds


def test_planning_actions_exclude_the_other_orientation():
    records = evaluate(data(), "leave_parent")
    expected = policy_actions(records)
    altered = copy.deepcopy(records)
    for row in altered:
        if row["fold"] == 1:
            row["admitted_mass"] = [1, 0, 0]
    np.testing.assert_array_equal(expected, policy_actions(altered))


def test_projection_uses_only_scoring_half_and_identical_actions_pair_exactly():
    states = data()
    actions = np.ones((16, 6, 3)) / 3
    counts = {task + "/" + role: 16 for task in ("math", "code") for role in ROLES}
    before = bootstrap(states, actions, 32, counts, draws=50)
    changed = copy.deepcopy(states)
    for state in changed:
        assignment = folds(state["parent"])
        for role in ROLES:
            state["panel"][role]["native"][:, :, assignment[role][0]] = 100
    after = bootstrap(changed, actions, 32, counts, draws=50)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(before[1][:, 0], before[1][:, 2])


def test_joint_power_requires_both_measurement_effects():
    rng = np.random.default_rng(42)
    point = np.array([-0.03, 0, 0.001, 0.001, 0.001, 0.01, 0.10])
    boot = point + rng.normal(0, 0.001, (1000, 7))
    assert conditional_power(point, boot, 0.75, 1.25)["measurement_joint_power"] > 0.99
    point[6] = -0.10
    boot[:, 6] -= 0.20
    assert conditional_power(point, boot, 0.75, 1.25)["measurement_joint_power"] == 0


def test_design_cannot_rename_measurement_as_method_efficacy_or_ignore_cost():
    row = {
        "parents": 48,
        "cost": {"base_plus_scoring_reservation_sus": 700},
        "sensitivity": [
            {
                "effect_shrinkage": 0.75,
                "sd_inflation": 1.25,
                "measurement_joint_power": 0.95,
                "admission_planning": {
                    "ridge_h1": {"joint_power_at_practical_effect": 0.99}
                },
            }
        ],
    }
    assert "not_method_efficacy" in choose_design([row], False, 966)["purpose"]
    assert (
        choose_design([row], True, 966)["purpose"]
        == "admission_confirmation_design_candidate"
    )
    assert choose_design([row], True, 750) is None
    row["cost"]["base_plus_scoring_reservation_sus"] = 809
    assert choose_design([row], True, 966) is None


def test_cost_requires_all_streams_and_counts_thirteen_checkpoints():
    counts = {task + "/" + role: 128 for task in ("math", "code") for role in ROLES}
    benchmarks = [
        {
            "task": {"stream": model + "/" + task},
            "model_load_seconds": 20,
            "observations": [
                {
                    "role": role,
                    "generation_seconds": 16,
                    "verification_seconds": 0,
                    "count": 16,
                    "restore_seconds": 1,
                }
                for role in ROLES
            ],
        }
        for model in ("llama", "qwen")
        for task in ("code", "math")
    ]
    limits = {row["task"]["stream"]: 3600 for row in benchmarks}
    result = price(48, counts, limits, benchmarks)
    assert result["base_plus_scoring_reservation_sus"] > 48
    assert not result["humaneval_directly_measured"]
    with pytest.raises(ValueError, match="four"):
        price(48, counts, limits, benchmarks[:3])
