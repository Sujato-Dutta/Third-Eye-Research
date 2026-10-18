"""Admission value must survive independent draws/items and parent separation."""

import copy

import numpy as np
import pytest

from third_eye.experiments.admission import (
    aggregate,
    arrays,
    conditional_item_bootstrap,
    effective_delta_norm,
    gate_review,
    planning,
    state_summary,
    synthetic_controls,
)
from third_eye.experiments.mechanism import ROLES, folds


def toy():
    parent = {
        "items": {
            role: [
                {"id": str(i), "prompt_sha256": f"{role}-{i}", "correct": i % 2}
                for i in range(16)
            ]
            for role in ROLES
        }
    }
    immediate = [copy.deepcopy(parent) for _ in range(3)]
    cube = [
        [[copy.deepcopy(parent) for _ in range(3)] for _ in range(3)] for _ in range(3)
    ]
    return parent, immediate, cube


def fill(artifact, value):
    for items in artifact["items"].values():
        for item in items:
            item["correct"] = value


def states():
    result = []
    for i in range(8):
        parent, immediate, cube = toy()
        fill(immediate[0], 1)
        for draw in range(3):
            fill(cube[0][0][draw], i % 2)
        panel = arrays(parent, immediate, cube)
        result.append(
            {
                "trajectory_id": str(i),
                "stream": f"s{i // 2}",
                "parent": parent,
                "panel": panel,
                "effective_update_norms": [1.0, 2.0, 3.0],
                "summary": state_summary(parent, panel),
            }
        )
    return result


def test_effective_norm_is_gauge_invariant_and_matches_actual_update():
    rng = np.random.default_rng(17)
    a0, a1 = rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
    b0, b1 = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    old, new = {"layer": (a0, b0)}, {"layer": (a1, b1)}
    expected = np.linalg.norm(2 * (b1 @ a1 - b0 @ a0))
    assert effective_delta_norm(old, new, 2) == pytest.approx(expected)
    gauged = {"layer": (a1 * 3, b1 / 3)}
    assert effective_delta_norm(old, gauged, 2) == pytest.approx(expected)
    assert effective_delta_norm(new, gauged, 2) == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError, match="Matched LoRA"):
        effective_delta_norm(old, {}, 2)


def test_proxy_is_not_a_clipped_noisy_oracle():
    parent, immediate, cube = toy()
    fill(immediate[0], 1)
    fill(cube[0][0][0], 1)
    fill(cube[0][0][1], 1)
    fill(cube[0][0][2], 0)
    result = state_summary(parent, arrays(parent, immediate, cube))
    # Selection on two positive draws admits the negative held-out branch.
    # The signed independently scored proxy must be allowed to lose value.
    assert result["metrics"]["proxy_minus_hold"] < 0
    assert result["metrics"]["noisy_positive_part_minus_independent_admission"] > 0
    assert result["same_outcome_selection_optimism_by_K"]["1"] == 0


def test_neutral_identity_has_no_artificial_admission_or_interaction_signal():
    parent, immediate, cube = toy()
    result = state_summary(parent, arrays(parent, immediate, cube))
    assert result["metrics"]["forced_minus_hold"] == 0
    assert result["metrics"]["proxy_minus_hold"] == 0
    assert result["metrics"]["proxy_admission_probability"] == 0
    assert result["interaction_agreement"] == 0
    broken = copy.deepcopy(cube)
    broken[0][0][0]["items"][ROLES[0]][0]["id"] = "changed"
    with pytest.raises(ValueError, match="identities"):
        arrays(parent, immediate, broken)


def test_scoring_items_cannot_change_fitted_gate_or_threshold():
    data = states()
    baseline = gate_review(data)
    changed = copy.deepcopy(data)
    for state in changed:
        assignment = folds(state["parent"])
        for role in ROLES:
            index = assignment[role][1]
            state["panel"][role]["cube"][:, :, :, index] *= -1
    altered = gate_review(changed)
    for before, after in zip(baseline, altered):
        if before["fold"] == 0:
            assert before["model"] == after["model"]
            assert before["threshold"] == after["threshold"]
            assert before["admitted_mass"] == after["admitted_mass"]
            assert before["test_index"] not in before["train_indices"]
            assert before["test_index"] != before["validation_index"]


def test_counterfactual_donors_never_train_a_gate():
    data = states()
    original = gate_review(data)
    changed = copy.deepcopy(data)
    for state in changed:
        for role in ROLES:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        state["panel"][role]["cube"][:, i, j, :] *= -1
    assert original == gate_review(changed)


def test_headroom_compares_with_one_population_constant_not_per_parent_oracle():
    data = states()
    for state in data:
        state["summary"]["metrics"]["forced_minus_hold"] = (
            0.04 if int(state["trajectory_id"]) % 2 else -0.04
        )
        state["summary"]["metrics"]["proxy_minus_hold"] = 0.02
    report = aggregate(data, gate_review(data))
    assert report["independent_parents"] == 8
    assert report["proxy_minus_better_constant"]["mean"] == pytest.approx(0.02)
    assert report["proxy_minus_better_constant"]["not_true_oracle_or_upper_bound"]
    assert report["automatic_expansion"] is False


def test_planning_is_sensitive_to_variance_not_candidate_count():
    effects = np.asarray(
        [[0.02 * (-1) ** i, 0.01 * (-1) ** (i // 2)] for i in range(8)]
    )
    report = planning(effects)
    rows = report["rows"]
    base = next(
        r
        for r in rows
        if r["parents"] == 48
        and r["sd_multiplier"] == 1
        and r["assumed_effect_each_contrast_pp"] == 0.5
    )
    noisy = next(
        r
        for r in rows
        if r["parents"] == 48
        and r["sd_multiplier"] == 2
        and r["assumed_effect_each_contrast_pp"] == 0.5
    )
    assert (
        base["joint_superiority_power_approx"] > noisy["joint_superiority_power_approx"]
    )
    assert noisy["conservative_joint_80pct_MDE_pp"] == pytest.approx(
        2 * base["conservative_joint_80pct_MDE_pp"]
    )
    assert planning(np.zeros((8, 2)))["status"].startswith("degenerate")


def test_synthetic_control_detects_known_signal_without_inventing_null_value():
    results = synthetic_controls()
    assert abs(results["equal_value_null"]["independent_selection_value"]) < 0.001
    assert results["equal_value_null"]["same_outcome_maximum"] > 0.01
    assert results["recoverable_candidate_signal"]["true_selected_value"] > 0.05
    assert results["recoverable_admission"]["independent_admission_value"] > 0.015
    assert results["not_real_data_power"] is True


def test_shared_scoring_items_do_not_become_independent_when_parents_repeat():
    data = states()
    fits = gate_review(data)
    single = conditional_item_bootstrap(
        data[:1], [r for r in fits if r["test_index"] == 0], draws=200
    )
    repeated = [copy.deepcopy(data[0]) for _ in range(8)]
    repeated_fits = []
    for i in range(8):
        for row in fits:
            if row["test_index"] == 0:
                copy_row = copy.deepcopy(row)
                copy_row["test_index"] = i
                repeated_fits.append(copy_row)
    result = conditional_item_bootstrap(repeated, repeated_fits, draws=200)
    assert result["distinct_role_half_pools"] == single["distinct_role_half_pools"]
    assert result["metrics"]["forced"]["bootstrap_sd"] == pytest.approx(
        single["metrics"]["forced"]["bootstrap_sd"]
    )
