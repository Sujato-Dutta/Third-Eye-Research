"""Fresh admission planning must fail closed and preserve independent scoring."""

import copy

import numpy as np
import pytest

from test_admission import fill, toy
from third_eye.experiments.admission import gate_review
from third_eye.experiments.admission_pilot import (
    native_arrays,
    native_summary,
    runtime_limits,
)
from third_eye.experiments.mechanism import ROLES, folds


def data():
    states = []
    for index in range(16):
        parent, immediate, cube = toy()
        fill(immediate[0], 1)
        for draw in range(3):
            fill(cube[0][0][draw], index % 2)
        native = [cube[k][k] for k in range(3)]
        states.append(
            {
                "trajectory_id": str(index),
                "stream": str(index // 4),
                "parent": parent,
                "panel": native_arrays(parent, immediate, native),
                "effective_update_norms": [1.0, 2.0, 3.0],
            }
        )
    return states


def test_native_panels_do_not_fabricate_counterfactuals_and_identity_is_zero():
    parent, immediate, cube = toy()
    panel = native_arrays(parent, immediate, [cube[k][k] for k in range(3)])
    assert all(set(v) == {"h1", "native"} for v in panel.values())
    assert panel[ROLES[0]]["native"].shape == (3, 3, 16)
    result = native_summary({"parent": parent, "panel": panel})
    assert result["forced_minus_hold"] == result["proxy_minus_hold"] == 0
    bad = copy.deepcopy(immediate)
    bad[0]["items"][ROLES[0]][0]["correct"] = True
    with pytest.raises(ValueError, match="binary"):
        native_arrays(parent, bad, [cube[k][k] for k in range(3)])


def test_runtime_guard_includes_every_parent_and_overhead():
    measured = {str(i): [3600.0, 3601.0] for i in range(4)}
    limits, reserved = runtime_limits(measured)
    assert set(limits.values()) == {5400}
    assert reserved == pytest.approx(24 + 1.8525)
    with pytest.raises(ValueError, match="exceeds B cap"):
        runtime_limits({str(i): [50_000.0, 60_000.0] for i in range(4)})
    with pytest.raises(ValueError, match="finite"):
        runtime_limits({str(i): [float("nan"), 1.0] for i in range(4)})


def test_sixteen_parent_gate_never_trains_on_held_scoring_items():
    states = data()
    baseline = gate_review(states, expected_parents=16, validation_stride=4)
    changed = copy.deepcopy(states)
    for state in changed:
        assignment = folds(state["parent"])
        for role in ROLES:
            state["panel"][role]["native"][:, :, assignment[role][1]] *= -1
    altered = gate_review(changed, expected_parents=16, validation_stride=4)
    assert len(baseline) == 64
    for before, after in zip(baseline, altered):
        assert before["test_index"] not in before["train_indices"]
        assert before["validation_index"] not in before["train_indices"]
        if before["fold"] == 0:
            assert before["model"] == after["model"]
            assert before["threshold"] == after["threshold"]
            np.testing.assert_array_equal(
                before["admitted_mass"], after["admitted_mass"]
            )
