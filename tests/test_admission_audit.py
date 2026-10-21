"""Decision diagnostics must separate context from features without scoring leakage."""

import copy

import numpy as np

from test_admission import fill, toy
from third_eye.experiments.admission import gate_review
from third_eye.experiments.admission_audit import (
    action_tensor,
    crossed_bootstrap,
    evaluate,
    prepare,
    reproduce,
    splits,
)
from third_eye.experiments.admission_pilot import native_arrays
from third_eye.experiments.mechanism import ROLES, folds


def data():
    result = []
    for group, stream in enumerate(
        ("llama/code", "llama/math", "qwen/code", "qwen/math")
    ):
        for seed in range(4):
            parent, immediate, cube = toy()
            fill(immediate[0], 1)
            for draw in range(3):
                fill(cube[0][0][draw], int(group >= 2))
            result.append(
                {
                    "trajectory_id": f"{group}-{seed}",
                    "seed": seed,
                    "stream": stream,
                    "parent": parent,
                    "panel": native_arrays(
                        parent, immediate, [cube[k][k] for k in range(3)]
                    ),
                    "effective_update_norms": [1.0, 2.0, 3.0],
                }
            )
    return result


def test_seed_sensitivity_excludes_entire_test_and_validation_blocks():
    prepared = prepare(data(), 0)
    for train, validation, test in splits(prepared, "leave_seed"):
        assert (len(train), len(validation), len(test)) == (8, 4, 4)
        assert set(train).isdisjoint(validation) and set(train).isdisjoint(test)
        assert {prepared[i]["seed"] for i in train}.isdisjoint(
            {prepared[i]["seed"] for i in validation + test}
        )


def test_context_rule_recovers_known_backbone_signal_without_candidate_features():
    rows = evaluate(data(), "leave_parent")
    selected = [r for r in rows if r["policy"] == "backbone_h2"]
    assert len(selected) == 32
    assert np.mean([r["metrics"]["minus_hold"] for r in selected]) == 0.25
    assert all(r["within_parent_rank_rho"] is None for r in selected)
    assert reproduce(
        gate_review(data(), expected_parents=16, validation_stride=4), rows
    )


def test_scoring_items_never_choose_a_model_or_threshold():
    states = data()
    original = evaluate(states, "leave_parent")
    changed = copy.deepcopy(states)
    for state in changed:
        assignment = folds(state["parent"])
        for role in ROLES:
            state["panel"][role]["native"][:, :, assignment[role][1]] *= -1
    altered = evaluate(changed, "leave_parent")
    for before, after in zip(original, altered):
        if before["fold"] == 0:
            assert (
                before["alpha"] == after["alpha"]
                and before["threshold"] == after["threshold"]
            )
            np.testing.assert_array_equal(before["prediction"], after["prediction"])
            np.testing.assert_array_equal(
                before["admitted_mass"], after["admitted_mass"]
            )


def test_test_parent_future_never_enters_its_own_prediction():
    states = data()
    original = evaluate(states, "leave_parent")
    changed = copy.deepcopy(states)
    for role in ROLES:
        changed[0]["panel"][role]["native"] *= -1
    altered = evaluate(changed, "leave_parent")
    for before, after in zip(original, altered):
        if before["test_index"] == 0:
            assert (
                before["alpha"] == after["alpha"]
                and before["threshold"] == after["threshold"]
            )
            np.testing.assert_array_equal(before["prediction"], after["prediction"])


def test_shared_item_resampling_preserves_identical_policies_and_item_noise():
    states = data()
    names, actions = action_tensor(states, evaluate(states, "leave_parent"))
    forced = names.index("forced")
    pair = np.stack([actions[forced], actions[forced]])
    point, boot, _, _ = crossed_bootstrap(states, pair, mode="item", draws=100)
    np.testing.assert_array_equal(boot[:, 0], boot[:, 1])
    assert boot[:, 0].std() > 0
    np.testing.assert_array_equal(point[0], point[1])
