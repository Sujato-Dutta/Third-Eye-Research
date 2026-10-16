"""Development review contracts use synthetic data only."""

import copy

import numpy as np
import pytest

from test_expected_forecasting import repeated_records
from third_eye.forecasting.review import (
    check_frozen_model,
    held_out_draws,
    mean_views,
    pilot_folds,
    ridge_prediction,
    run_review,
)


def fixtures():
    rows = [r for r in repeated_records(8) if r["generation"] == 0]
    for r in rows:
        r["utility_weights"] = [1 / 3] * 3
        r["continuation_replicates"] += [
            dict(r["continuation_replicates"][1], seed=s) for s in (14, 15)
        ]
    return rows


def test_whole_trajectory_folds_have_unique_heldout_coverage():
    rows = fixtures()
    tested = []
    for parts in pilot_folds(rows):
        ids = [
            {r["trajectory_id"] for r in parts[p]}
            for p in ("train", "validation", "test")
        ]
        assert list(map(len, ids)) == [6, 1, 1]
        assert all(not ids[i] & ids[j] for i in range(3) for j in range(i))
        tested += list(ids[2])
    assert len(set(tested)) == 8
    with pytest.raises(ValueError, match="eight"):
        list(pilot_folds(rows[:21]))


def test_ridge_never_selects_on_test_outcomes():
    parts = next(pilot_folds(fixtures()))
    changed = copy.deepcopy(parts)
    for row in changed["test"]:
        row["labels"]["h1"] = dict.fromkeys(("target", "ood", "retention"), 0.5)
        for draw in row["continuation_replicates"]:
            draw["h2"] = dict.fromkeys(("target", "ood", "retention"), -0.5)
    for horizon in (1, 2):
        for simple in (True, False):
            before, a = ridge_prediction(parts, horizon, simple)
            after, b = ridge_prediction(changed, horizon, simple)
            np.testing.assert_array_equal(before, after)
            assert a == b


def test_native_labels_preserved_and_draws_really_disjoint():
    rows = fixtures()
    before = copy.deepcopy(rows)
    mean_views(rows)
    assert rows == before
    report = held_out_draws(rows[:3], np.asarray([1 / 3] * 3))
    assert len(report["rotations"]) == 10
    for rotation in report["rotations"]:
        selected, scoring = (
            set(rotation["selection_draw_indices"]),
            set(rotation["scoring_draw_indices"]),
        )
        assert not selected & scoring
        assert selected | scoring == set(range(5))


def test_complete_synthetic_review_stays_development_only(tmp_path):
    rows = fixtures()
    streams = {r["state_id"]: "fixture" for r in rows}

    class FrozenFixture:
        metadata = {
            "kind": "expected",
            "utility_weights": [1 / 3] * 3,
            "selection_trajectory_ids": ["old"],
            "selection_state_ids": ["old"],
            "weights_sha256": "a" * 64,
        }

        def predict_distribution(self, views):
            assert all(
                set(v) == {"state_id", "candidate_id", "precommit"} for v in views
            )
            return {
                "mean": np.full((len(views), 3), 0.1),
                "h1": np.zeros((len(views), 3)),
                "utility_std": np.zeros(len(views)),
            }

    fixture = FrozenFixture()
    check_frozen_model(rows, fixture)
    fixture.metadata = dict(fixture.metadata, selection_state_ids=[rows[0]["state_id"]])
    with pytest.raises(ValueError, match="developed"):
        check_frozen_model(rows, fixture)
    fixture.metadata = dict(fixture.metadata, selection_state_ids=["old"])
    report = run_review(
        rows, tmp_path / "review", streams, epochs=2, seeds=(42,), frozen_model=fixture
    )
    assert len(report["states"]) == 8
    assert report["development_only"] is True
    assert report["gate2_passed"] is None
    assert report["automatic_expansion"] is False
    assert report["primary_comparison"] == "frozen_expected_minus_observed_h1"
    assert all(np.isfinite(list(s["values"].values())).all() for s in report["states"])
