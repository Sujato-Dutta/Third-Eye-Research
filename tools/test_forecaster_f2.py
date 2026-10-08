"""Synthetic F2 contracts, not empirical forecasting evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from extensions.forecasting_f2.model import build_model
from extensions.forecasting_f2.training import (
    ContrastForecaster,
    group_inputs,
    objective,
    train,
    vector_loss,
)
from extensions.forecasting_f2.inference import registered_loader
from third_eye.forecasting.training import Forecaster

spec = importlib.util.spec_from_file_location(
    "forecast_fixture_f2", ROOT / "tests/test_forecasting.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def records():
    rows = fixture.records(6)
    for r in rows:
        r["protocol_amendment"] = "A2"
    return rows


def tensors(rows):
    arrays, _ = group_inputs(rows)
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]


def test_set_permutation_and_batch_permutation_equivariance():
    torch.manual_seed(1)
    model = build_model().eval()
    inputs = tensors(records())
    before = model(*inputs)
    permutation = [2, 0, 1]
    after = model(*(a[:, permutation] for a in inputs))
    assert torch.allclose(
        after["prediction"], before["prediction"][:, permutation], atol=1e-6
    )
    assert torch.allclose(
        model(*(a.flip(0) for a in inputs))["prediction"],
        before["prediction"].flip(0),
        atol=1e-6,
    )
    assert torch.allclose(
        before["advantage_h1"].sum(1),
        torch.zeros_like(before["advantage_h1"][:, 0]),
        atol=1e-6,
    )
    assert torch.allclose(
        before["advantage_continuation"].sum(1),
        torch.zeros_like(before["advantage_continuation"][:, 0]),
        atol=1e-6,
    )


def test_no_set_control_has_independent_candidate_predictions():
    model = build_model("no_set").eval()
    original = tensors(records())
    changed = [a.clone() for a in original]
    changed[1][:, 0] += 2
    assert torch.equal(
        model(*original)["prediction"][:, 1:], model(*changed)["prediction"][:, 1:]
    )


def test_postcommit_labels_runtime_and_model_identifiers_are_excluded():
    rows = records()
    changed = copy.deepcopy(rows)
    for r in changed:
        r.update(
            labels={"secret": 99999},
            continuation_available=0,
            runtime={"seconds": 1e8},
            model="different",
            seed=999,
        )
        r["precommit"]["unexpected_future_metrics"] = [99999]
        r["state_id"] = "renamed_" + r["state_id"]
        r["candidate_id"] = "renamed_" + r["candidate_id"]
    assert all(
        np.array_equal(a, b)
        for a, b in zip(group_inputs(rows)[0], group_inputs(changed)[0])
    )


def test_reject_incomplete_and_mismatched_parent_sets():
    rows = records()
    with pytest.raises(ValueError, match="three unique"):
        group_inputs(rows[1:])
    bad = copy.deepcopy(rows)
    bad[0]["precommit"]["state"]["target"] += 0.1
    with pytest.raises(ValueError, match="parent context"):
        group_inputs(bad)


def test_relative_loss_and_tied_labels_are_finite():
    x = torch.randn(2, 3, 3, requires_grad=True)
    y = torch.randn_like(x)
    offset = torch.randn(2, 1, 3)
    assert torch.allclose(
        vector_loss(x + offset, y + offset), vector_loss(x, y), atol=1e-6
    )
    zero = torch.zeros_like(x)
    out = {"prediction": x, "h1": x / 2, "continuation": x / 2}
    loss = objective(
        out, zero, zero, torch.tensor([1 / 3] * 3), torch.ones(3), 0.01, 2, "full"
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_train_save_reload_selection_and_checksum(tmp_path):
    rows = records()
    train_rows = [r for r in rows if r["trajectory_id"] in {"t0", "t1", "t2", "t3"}]
    val_rows = [r for r in rows if r["trajectory_id"] in {"t4", "t5"}]
    f = train(train_rows, val_rows, tmp_path / "fit", epochs=8, patience=8)
    reloaded = ContrastForecaster.load(tmp_path / "fit")
    assert np.array_equal(f.predict(val_rows), reloaded.predict(val_rows))
    assert f.select(val_rows[:3]) == reloaded.select(val_rows[:3])
    with registered_loader():
        dispatched = Forecaster.load(tmp_path / "fit")
        assert np.array_equal(dispatched.predict(val_rows), f.predict(val_rows))
    assert "test" not in json.loads((tmp_path / "fit/metrics.json").read_text())
    assert f.metadata["test_evaluated"] is False
    with pytest.raises(ValueError, match="one shared-parent"):
        f.select(val_rows)
    with (tmp_path / "fit/weights.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ContrastForecaster.load(tmp_path / "fit")


def test_h1_training_does_not_use_h2_supervision(tmp_path):
    rows = records()
    train_rows, val_rows = rows[:24], rows[24:]
    changed_train, changed_val = copy.deepcopy(train_rows), copy.deepcopy(val_rows)
    for r in changed_train + changed_val:
        r["labels"]["h2"] = {"target": 0.8, "ood": -0.8, "retention": 0.7}
    f = train(
        train_rows, val_rows, tmp_path / "original", horizon=1, epochs=3, patience=3
    )
    g = train(
        changed_train,
        changed_val,
        tmp_path / "changed_h2",
        horizon=1,
        epochs=3,
        patience=3,
    )
    assert np.array_equal(f.predict(val_rows), g.predict(val_rows))
    assert all(
        torch.equal(a, b) for a, b in zip(f.model.parameters(), g.model.parameters())
    )


def test_no_trajectory_leakage_and_parameter_ceiling(tmp_path):
    rows = records()
    with pytest.raises(ValueError, match="leakage"):
        train(rows, rows, tmp_path / "bad", epochs=1)
    assert sum(p.numel() for p in build_model().parameters()) < 1_000_000
    assert sum(p.numel() for p in build_model("no_set").parameters()) == sum(
        p.numel() for p in build_model().parameters()
    )


def test_f2_connects_to_frozen_online_runner_without_h2_harvesting(
    tmp_path, monkeypatch
):
    from third_eye.experiments.online import run_online
    from third_eye.evaluation.runner import Consequences

    label_spec = importlib.util.spec_from_file_location(
        "label_fixture_f2", ROOT / "tests/test_labeling.py"
    )
    label_fixture = importlib.util.module_from_spec(label_spec)
    label_spec.loader.exec_module(label_fixture)
    rows = records()
    f = train(rows[:24], rows[24:], tmp_path / "forecaster", epochs=2, patience=2)
    labeler = label_fixture.setup_labeler(tmp_path / "online", monkeypatch)
    monkeypatch.setattr(
        "third_eye.experiments.online.evaluate",
        lambda *args: Consequences(0.4, 0.2, 0.6),
    )
    result = run_online(
        labeler.backend,
        labeler.config,
        labeler.splits,
        labeler.verifier,
        tmp_path / "online",
        labeler.manifest_hash,
        "direct",
        forecaster=f,
    )
    assert result["complete"] and len(labeler.backend.starts) == 1
    forecast = json.loads(
        (tmp_path / "online/online/generation_0/forecast.json").read_text()
    )
    assert np.asarray(forecast).shape == (3, 3)
    assert not (tmp_path / "online/meta_labels.jsonl").exists()
