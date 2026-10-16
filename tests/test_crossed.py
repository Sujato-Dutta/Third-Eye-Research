"""Mechanism identification, independent scoring and exact rollback controls."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from third_eye.data.schema import Correction, Example
from third_eye.experiments.crossed import cells, decompose, state_review
from third_eye.io import digest
from third_eye.updates.candidates import batch_hash


def cli():
    import sys

    experiments = Path(__file__).resolve().parents[1] / "experiments"
    sys.path.insert(0, str(experiments))
    spec = importlib.util.spec_from_file_location(
        "crossed_control_cli", experiments / "crossed_corrections.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel():
    parent = {
        "items": {
            r: [dict(id=str(i), prompt_sha256=f"{r}-{i}", correct=0) for i in range(8)]
            for r in ("target_dev", "ood_dev", "retention_dev")
        }
    }
    immediate = [copy.deepcopy(parent) for _ in range(3)]
    cube = [
        [[copy.deepcopy(parent) for _ in range(3)] for _ in range(3)] for _ in range(3)
    ]
    return parent, immediate, cube


def fill(artifact, correct):
    for items in artifact["items"].values():
        for item in items:
            item["correct"] = correct


def test_crossing_separates_recipient_donor_and_interaction():
    additive = np.array([1, 3, -2])[:, None] + np.array([0.1, 0.3, 0.9])[None, :]
    result = decompose(additive)
    assert result["native_pairing_surplus"] == pytest.approx(0)
    assert result["interaction_mean_square"] == pytest.approx(0)
    paired = decompose(additive + np.eye(3))
    assert paired["native_pairing_surplus"] == pytest.approx(2 / 3)
    assert paired["interaction_mean_square"] > 0
    assert len(cells()) == len(set(cells())) == 27
    with pytest.raises(ValueError, match="Finite complete"):
        decompose([[float("nan")] * 3] * 3)


def test_pairing_contrast_removes_immediate_score_and_marginal_effects():
    parent, m1, cube = panel()
    for i in range(3):
        for s in range(3):
            fill(cube[i][i][s], 1)
    result = state_review(parent, m1, cube)
    assert result["comparison"]["native_pairing_surplus"] == pytest.approx(2 / 3)
    fill(m1[0], 1)
    changed = state_review(parent, m1, cube)
    assert (
        changed["comparison"]["native_pairing_surplus"]
        == result["comparison"]["native_pairing_surplus"]
    )
    assert changed["fold_decomposition"][0]["total_future"][
        "native_pairing_surplus"
    ] == pytest.approx(
        changed["fold_decomposition"][0]["continuation_gain"]["native_pairing_surplus"]
    )


def test_scoring_draw_cannot_change_lookahead_selection():
    parent, m1, cube = panel()
    # Native selector prefers row0. Uniform-donor selector prefers row1.
    for s in (0, 1):
        fill(cube[0][0][s], 1)
        fill(cube[1][0][s], 1)
        fill(cube[1][2][s], 1)
    fill(cube[1][1][2], 1)
    first = state_review(parent, m1, cube)["comparison"]
    assert first["common_donor_selection_minus_native"] == 1
    fill(cube[1][1][2], 0)
    fill(cube[0][0][2], 1)
    second = state_review(parent, m1, cube)["comparison"]
    assert second["common_donor_selection_minus_native"] == -1
    assert (
        second["common_native_choice_disagreement"]
        == first["common_native_choice_disagreement"]
        == 1
    )
    broken = copy.deepcopy(cube)
    broken[0][0][0]["items"]["target_dev"][0]["id"] = "wrong"
    with pytest.raises(ValueError, match="items/order"):
        state_review(parent, m1, broken)


def test_saved_batch_requires_verified_pool_and_exact_train_identity():
    module = cli()
    ex = Example("train0", "problem", "answer", "train")
    correction = Correction(ex, "verified answer", 1)
    batch = [correction.to_dict()]
    pool = {"items": batch}
    splits = {"train": [ex]}
    sha = batch_hash([correction])
    assert module.validate_batch(batch, pool, splits, 1, 1, sha) == (correction,)
    with pytest.raises(ValueError, match="unique"):
        module.validate_batch(batch * 2, pool, splits, 2, 1, sha)
    with pytest.raises(ValueError, match="verified train"):
        module.validate_batch(batch, {"items": []}, splits, 1, 1, sha)
    changed = copy.deepcopy(batch)
    changed[0]["example"]["prompt"] = "different problem"
    with pytest.raises(ValueError, match="verified train"):
        module.validate_batch(
            changed,
            {"items": changed},
            splits,
            1,
            1,
            batch_hash(
                [Correction(Example(**changed[0]["example"]), "verified answer", 1)]
            ),
        )
    assert module.validate_batch([], pool, splits, 1, 0, None) == ()


def test_historical_scores_cannot_silently_fill_the_corrected_matrix(monkeypatch):
    module = cli()
    cfg = SimpleNamespace(model=SimpleNamespace(chat_kwargs={}))
    monkeypatch.setattr(module, "item_artifact", lambda *a: {"adapter_hash": "m1"})
    with pytest.raises(ValueError, match="Historical or mismatched"):
        module.fixed_items("old", "m1", {}, cfg)
    artifact = {
        "input_rendering": {
            "chat_kwargs": {"date_string": "11 Oct 2026"},
            "template_sha256": "a" * 64,
        }
    }
    monkeypatch.setattr(module, "item_artifact", lambda *a: artifact)
    assert module.fixed_items("new", "m1", {}, cfg) == artifact


def test_failed_native_replay_rolls_back_and_cannot_publish_cell(tmp_path, monkeypatch):
    module = cli()
    example = Example("a", "p", "a", "train")
    batch = [Correction(example, "a", 1)]
    draw = {
        "seed": 50,
        "continuation_adapter_hash": "expected",
        "batch_hash": digest("batch"),
    }
    monkeypatch.setattr(module, "donor_input", lambda *a: (draw, batch, []))

    class Backend:
        state = "initial"

        def load_checkpoint(self, path):
            self.state = "m1"

        def state_hash(self):
            return self.state

        def snapshot(self):
            return self.state

        def train(self, batch, seed, log_path):
            self.state = "wrong-m2"
            return {"optimizer_steps": 50}

        def restore(self, value):
            self.state = value

    backend = Backend()
    cfg = SimpleNamespace(training=SimpleNamespace(max_steps=50))
    recipient = {"checkpoints": {"t1": "saved"}, "candidate_adapter_hash": "m1"}
    with pytest.raises(RuntimeError, match="Native diagonal replay"):
        module.execute_cell(
            backend,
            cfg,
            {},
            None,
            recipient,
            recipient,
            50,
            tmp_path / "cell",
            diagonal=True,
        )
    assert backend.state_hash() == "m1"
    assert not (tmp_path / "cell/cell.json").exists()


def test_terminal_transfer_preserves_recipient_not_donor(tmp_path, monkeypatch):
    module = cli()
    draw = {"seed": 50, "batch_hash": None}
    monkeypatch.setattr(module, "donor_input", lambda *a: (draw, (), []))
    monkeypatch.setattr(
        module, "item_artifact", lambda path, sha, splits: {"checked": sha}
    )
    items = tmp_path / "m1.json"
    items.write_text("{}")

    class Backend:
        def load_checkpoint(self, path):
            pass

        def state_hash(self):
            return "recipient-m1"

        def snapshot(self):
            return "recipient-m1"

        def restore(self, value):
            assert value == "recipient-m1"

    recipient = {
        "checkpoints": {"t1": "saved"},
        "candidate_id": "recipient",
        "candidate_adapter_hash": "recipient-m1",
        "evaluation_artifacts": {"t1": str(items)},
    }
    result = module.execute_cell(
        Backend(),
        SimpleNamespace(),
        {},
        None,
        recipient,
        {"candidate_id": "donor"},
        50,
        tmp_path / "cell",
    )
    assert result["continuation_adapter_hash"] == "recipient-m1"
    assert (
        result["training"]["optimizer_steps"] == result["continuation_available"] == 0
    )
    assert result["items_path"] == str(items)
