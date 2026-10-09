"""Tabular comparator retains allowlisted pre-commit information only."""

import copy
import importlib.util
from pathlib import Path
import numpy as np
import empirical_comparator_v1 as comparator

spec = importlib.util.spec_from_file_location(
    "comparator_fixture",
    Path(__file__).resolve().parents[1] / "tests/test_forecasting.py",
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_post_update_fields_cannot_enter_comparator_features():
    rows = fixture.records(6)
    baseline = comparator.flat(rows)
    other = copy.deepcopy(rows)
    for r in other:
        r["runtime"] = {"future_loss": 999}
        r["labels"] = {
            "h1": {h: 999 for h in comparator.study.HEADS},
            "h2": {h: -999 for h in comparator.study.HEADS},
        }
        r["checkpoints"] = {"t2": "leak"}
    assert np.array_equal(baseline, comparator.flat(other))


def test_synthetic_controls_never_modify_observed_labels():
    rows = fixture.records(8)
    before = copy.deepcopy(rows)
    train = rows[:30]
    val = rows[30:]
    a, b = comparator.synthetic_rows(train, val, 0.0, 42)
    assert rows == before
    assert a != train and b != val
    for rs in [a, b]:
        for g in comparator.study.groups(rs):
            y = comparator.study.targets(rs, "future")[g]
            assert np.allclose(y.mean(0), 0)
