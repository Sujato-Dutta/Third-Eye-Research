"""Synthetic scientific-analysis checks; no real research fitting locally."""

import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import diagnose_forecast_signal_v1 as study

spec = importlib.util.spec_from_file_location(
    "signal_fixture", Path(__file__).resolve().parents[1] / "tests/test_forecasting.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def records():
    rows = fixture.records(6)
    for r in rows:
        r["analysis_stream"] = "synthetic/math"
        r["precommit"].setdefault("generation", r["generation"])
    return rows


def test_continuation_is_exact_increment_and_not_future_itself():
    rows = records()
    assert np.allclose(
        study.targets(rows, "immediate") + study.targets(rows, "continuation"),
        study.targets(rows, "future"),
    )
    with pytest.raises(ValueError):
        study.targets(rows, "test")


def test_decomposition_identity_regret_ties_and_terminal():
    rows = records()
    for g in study.groups(rows):
        for j, i in enumerate(g):
            rows[i]["labels"]["h1"] = dict(zip(study.HEADS, [0.1 * j] * 3))
            rows[i]["labels"]["h2"] = dict(zip(study.HEADS, [0.2 - 0.1 * j] * 3))
            rows[i]["continuation_available"] = 1
    r = study.decomposition(rows)
    assert r["metrics"]["strict_winner_reversal"]["state_mean"] == 1
    assert np.isclose(r["metrics"]["immediate_oracle_future_regret"]["state_mean"], 0.2)
    for row in rows:
        row["labels"]["h2"] = {h: 0.1 for h in study.HEADS}
    r = study.decomposition(rows)
    assert r["metrics"]["exact_winner_tie"]["state_mean"] == 1
    assert r["metrics"]["strict_winner_reversal"]["state_mean"] == 0


def test_nested_whole_trajectory_subsets_and_partition_guard():
    rows = records()
    a, b = study.subset(rows, 2), study.subset(rows, 4)
    assert {r["trajectory_id"] for r in a} <= {r["trajectory_id"] for r in b}
    assert len(study.groups(a)) == 4
    with pytest.raises(ValueError):
        study.assert_disjoint(a, b)
    with pytest.raises(ValueError):
        study.subset(rows, 7)


def test_ridge_uses_train_only_and_does_not_consume_validation_targets():
    rows = records()
    train, val = (
        study.subset(rows, 4),
        [r for r in rows if r not in study.subset(rows, 4)],
    )
    changed = copy.deepcopy(val)
    for r in changed:
        r["labels"] = {
            "h1": {h: 999 for h in study.HEADS},
            "h2": {h: -999 for h in study.HEADS},
        }
    for relative in [False, True]:
        assert np.array_equal(
            study.ridge(train, val, "future", relative),
            study.ridge(train, changed, "future", relative),
        )


def test_cluster_interval_counts_trajectories_and_undefined_values():
    rows = [r for r in records() if r["generation"] == 0]
    for g in study.groups(rows)[:3]:
        for i in g:
            rows[i]["trajectory_id"] = "shared"
    report = study.interval([0, 0, 0, 1, None, 1], rows)
    assert report["states"] == 5 and report["trajectories"] == 3
    assert np.isclose(report["trajectory_mean"], 2 / 3)
    with pytest.raises(ValueError):
        study.interval([0], rows)


def test_synthetic_gru_fit_reloads_and_labels_are_not_mutated(tmp_path):
    rows = records()
    train = study.subset(rows, 4)
    val = [r for r in rows if r not in train]
    before = copy.deepcopy(rows)
    pred = study.fit_gru(train, val, "continuation", 42, tmp_path / "fit")
    assert pred.shape == (len(val), 3) and np.isfinite(pred).all()
    assert rows == before


def test_fit_grid_and_no_candidate_order_artifact():
    tasks = study.specifications()
    assert len(tasks) == 36 and {t["seed"] for t in tasks} == {42, 43, 44}
    rows = records()
    y = study.targets(rows, "future")
    before = study.scores(rows, y, np.zeros_like(y))
    order = np.arange(len(rows))[::-1]
    after = study.scores([rows[i] for i in order], y[order], np.zeros_like(y))
    assert before["utility"]["top1"] == after["utility"]["top1"]
