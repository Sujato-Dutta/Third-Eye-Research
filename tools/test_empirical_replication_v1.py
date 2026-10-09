"""Synthetic checks of saved ridge replay and balanced trajectory uncertainty."""

import importlib.util
from pathlib import Path
import numpy as np
import empirical_replication_v1 as replication

spec = importlib.util.spec_from_file_location(
    "replication_fixture",
    Path(__file__).resolve().parents[1] / "tests/test_forecasting.py",
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_ridge_saved_coefficients_replay_original_method_exactly():
    rows = fixture.records(8)
    train = rows[:30]
    val = rows[30:]
    for target in replication.study.TARGETS:
        for relative in [False, True]:
            meta = replication.ridge_fit(train, target, relative)
            predicted = replication.ridge_predict(meta, val)
            assert np.array_equal(
                predicted, replication.study.ridge(train, val, target, relative)
            )


def test_stratified_bootstrap_weights_streams_equally():
    rows = fixture.records(4)
    for r in rows:
        r["analysis_stream"] = "a" if r["trajectory_id"] == "t0" else "b"
    values = [
        1.0 if rows[g[0]]["analysis_stream"] == "a" else 0.0
        for g in replication.study.groups(rows)
    ]
    result = replication.stratified_interval(values, rows)
    assert result["mean"] == 0.5 and result["ci95"] == [0.5, 0.5]
    assert result["covered_trajectories"] == 4 and result["covered_streams"] == 2


def test_random_ties_and_future_oracle_selection_have_correct_values():
    rows = fixture.records(4)
    for r in rows:
        r["analysis_stream"] = "synthetic"
    random = replication.future_selection(rows, np.zeros((len(rows), 3)))
    assert abs(random["greedy_policy_minus_random"]["mean"]) < 1e-12
    oracle = replication.future_selection(
        rows, replication.study.targets(rows, "future")
    )
    assert abs(oracle["forced_future_oracle_regret"]["mean"]) < 1e-12
