"""Synthetic held-out isolation and saved-prediction equivalence."""

import numpy as np
import pytest
import confirm_forecast_signal_v1 as confirm
import test_diagnose_forecast_signal_v1 as fixture


def test_heldout_guards_empty_shared_trajectories_and_states():
    rows = fixture.records()
    train = confirm.study.subset(rows, 4)
    test = [r for r in rows if r not in train]
    confirm.assert_held_out(train, test)
    with pytest.raises(ValueError):
        confirm.assert_held_out(train, [])
    with pytest.raises(ValueError):
        confirm.assert_held_out(train, train)
    with pytest.raises(ValueError):
        confirm.assert_held_out(train, [dict(test[0], state_id=train[0]["state_id"])])


def test_synthetic_predict_matches_original_fit_and_rejects_tampered_weights(tmp_path):
    rows = fixture.records()
    train = confirm.study.subset(rows, 4)
    test = [r for r in rows if r not in train]
    folder = tmp_path / "fit"
    pred = confirm.study.fit_gru(train, test, "future", 42, folder)
    assert np.array_equal(pred, confirm.predict(folder, test))
    (folder / "diagnostic_weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        confirm.predict(folder, test)
