"""Synthetic contracts for repeated targets, relative inputs and honest scoring."""

import copy
import json

import numpy as np
import pytest
import torch

from test_forecasting import records
from test_labeling import setup_labeler
from third_eye.forecasting.continuation import (
    candidate_inputs,
    repeated_targets,
    reliability_audit,
)
from third_eye.forecasting.dataset import split_records
from third_eye.forecasting.encoding import FEATURE_NAMES, labels
from third_eye.forecasting.scoring import independent_selection_report
from third_eye.forecasting.training import Forecaster, train_forecaster


def repeated_records(trajectories=9):
    data = records(trajectories)
    for row in data:
        row["candidate_adapter_hash"] = f"m1-{row['state_id']}-{row['candidate_id']}"
        row["continuation_replicates"] = [
            {
                "seed": seed,
                "candidate_adapter_hash": row["candidate_adapter_hash"],
                "continuation_adapter_hash": f"m2-{seed}",
                "evaluation_manifest_hash": row["manifest_hash"],
                "continuation_available": 1,
                "optimizer_steps": 50,
                "reserved_optimizer_steps": 50,
                "h2": dict(
                    zip(
                        ("target", "ood", "retention"),
                        (labels(row, 2) + offset).tolist(),
                    )
                ),
            }
            for seed, offset in ((11, -0.03), (12, 0), (13, 0.03))
        ]
    return data


def test_repeat_mean_and_covariance_preserve_shared_head_variation():
    rows = repeated_records(3)
    mean, covariance, draws = repeated_targets(rows)
    np.testing.assert_allclose(mean, np.stack([labels(r, 2) for r in rows]), atol=1e-16)
    np.testing.assert_allclose(covariance[0], np.full((3, 3), 0.0009))
    assert len(draws[0]) == 3
    audit = reliability_audit(rows)
    # Common-mode continuation shifts cancel in candidate differences.
    assert audit["states"][0]["mean_measurement_variance"] == pytest.approx(
        0, abs=1e-20
    )


@pytest.mark.parametrize(
    "defect", ["single", "seed", "m1", "manifest", "budget", "grid", "terminal"]
)
def test_invalid_distribution_targets_are_rejected(defect):
    rows = repeated_records(3)
    draw = rows[0]["continuation_replicates"][0]
    if defect == "single":
        rows[0]["continuation_replicates"] = [draw]
    elif defect == "seed":
        draw["seed"] = 12
    elif defect == "m1":
        draw["candidate_adapter_hash"] = "wrong"
    elif defect == "manifest":
        draw["evaluation_manifest_hash"] = "wrong"
    elif defect == "budget":
        draw["optimizer_steps"] = 49
    elif defect == "grid":
        draw["seed"] = 100
    else:
        draw["continuation_available"] = 0
        draw["optimizer_steps"] = 0  # M2 still differs from M1.
    with pytest.raises(ValueError):
        repeated_targets(rows)


def test_true_terminal_draws_have_no_fabricated_variation():
    rows = repeated_records(3)
    for row in rows:
        for draw in row["continuation_replicates"]:
            draw.update(
                continuation_available=0,
                optimizer_steps=0,
                continuation_adapter_hash=row["candidate_adapter_hash"],
                h2=row["labels"]["h1"],
            )
    mean, cov, _ = repeated_targets(rows)
    np.testing.assert_allclose(mean, np.stack([labels(r, 1) for r in rows]))
    np.testing.assert_allclose(cov, 0, atol=1e-30)


def test_relative_encoder_is_set_local_masked_and_outcome_blind():
    rows = repeated_records(3)
    before = candidate_inputs(rows)
    changed = copy.deepcopy(rows)
    for r in changed:
        r["labels"] = {"private": 999}
        r["continuation_replicates"] = [{"private": 999}]
    np.testing.assert_array_equal(candidate_inputs(changed), before)
    n = len(FEATURE_NAMES)
    feature = FEATURE_NAMES.index("pre_update_nll")
    np.testing.assert_allclose(before[:3, 3 + 2 * n + feature], [-1, 0, 1])
    changed = copy.deepcopy(rows)
    del changed[0]["precommit"]["features"]["pre_update_nll"]
    after = candidate_inputs(changed)
    np.testing.assert_allclose(after[:3, 3 + 2 * n + feature], [0, -0.5, 0.5])
    np.testing.assert_array_equal(after[3:], before[3:])
    with pytest.raises(ValueError, match="K=3"):
        candidate_inputs(rows[:2])


def test_expected_fit_checkpoint_and_untouched_test_normalization(tmp_path):
    torch.set_num_threads(1)
    parts = split_records(repeated_records())
    alternate = copy.deepcopy(parts)
    for row in alternate["test"]:
        row["precommit"]["features"]["pre_update_nll"] = 99999
        for draw in row["continuation_replicates"]:
            draw["h2"] = {h: 0.8 for h in ("target", "ood", "retention")}
    fits = [
        train_forecaster(
            p, tmp_path / str(i), kind="expected", epochs=3, patience=2, hidden=8
        )
        for i, p in enumerate((parts, alternate))
    ]
    assert fits[0].metadata["mean"] == fits[1].metadata["mean"]
    for name, tensor in fits[0].model.state_dict().items():
        torch.testing.assert_close(
            tensor, fits[1].model.state_dict()[name], rtol=0, atol=0
        )
    views = [
        {k: r[k] for k in ("state_id", "candidate_id", "precommit")}
        for r in parts["test"]
    ]
    loaded = Forecaster.load(tmp_path / "0")
    np.testing.assert_array_equal(fits[0].predict(views), loaded.predict(views))
    result = loaded.predict_distribution(views)
    np.testing.assert_allclose(
        result["mean"], result["h1"] + result["continuation_mean"]
    )
    assert (np.linalg.eigvalsh(result["covariance"]) > 0).all()
    assert np.isfinite(result["utility_std"]).all()
    assert loaded.metadata["trainable_parameters"] < 1_000_000
    metrics = json.loads((tmp_path / "0/metrics.json").read_text())
    assert metrics["test"]["draw_variance_mae"] >= 0
    with pytest.raises(ValueError, match="single H2"):
        train_forecaster(
            split_records(records()), tmp_path / "no_repeats", kind="expected"
        )


def test_risk_decision_abstains_and_never_reads_future_outcomes(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    fit = train_forecaster(
        split_records(repeated_records()),
        tmp_path / "fit",
        kind="expected",
        epochs=1,
        hidden=8,
        risk_penalty=2,
        allow_abstain=True,
    )
    views = repeated_records()[:3]
    distribution = {
        "mean": np.full((3, 3), 0.01),
        "h1": np.zeros((3, 3)),
        "continuation_mean": np.full((3, 3), 0.01),
        "covariance": np.tile(np.eye(3), (3, 1, 1)),
        "utility_std": np.full(3, 0.03),
    }
    monkeypatch.setattr(fit, "predict_distribution", lambda views: distribution)
    assert fit.select(views) is None
    assert fit.decision(views)["abstained"]
    fit.metadata["decision"]["allow_abstain"] = False
    assert fit.select(views) in (0, 1, 2)
    assert fit.select(views) == fit.select(views)


def test_collector_reuses_m1_restores_parent_and_commits_only_m1(tmp_path, monkeypatch):
    labeler = setup_labeler(tmp_path, monkeypatch)
    labeler.continuation_seed_offsets = (50_000_000, 100_000_000)
    native = labeler.label_state(0)
    assert len(labeler.backend.starts) == 12  # K*(one M1 + three M2).
    assert labeler.backend.state == ()
    assert all(labeler.backend.starts[i] == () for i in (0, 4, 8))
    for i in (0, 4, 8):
        assert (
            labeler.backend.starts[i + 1]
            == labeler.backend.starts[i + 2]
            == labeler.backend.starts[i + 3]
        )
    repeated_targets(native)
    assert len((tmp_path / "meta_labels.jsonl").read_text().splitlines()) == 3
    assert all(len(r["continuation_replicates"]) == 3 for r in native)


def test_repeat_failure_publishes_no_partial_state(tmp_path, monkeypatch):
    labeler = setup_labeler(tmp_path, monkeypatch, fail_call=3)
    labeler.continuation_seed_offsets = (50_000_000, 100_000_000)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        labeler.label_state(0)
    assert labeler.backend.state == ()
    assert not (tmp_path / "meta_labels.jsonl").exists()


def test_independent_score_uses_fresh_draws_and_clusters_trajectories():
    rows = repeated_records(3)
    decisions = {}
    for row in rows:
        row.update(
            selection_h1=row["labels"]["h1"],
            selection_items=["a"],
            scoring_items=["b"],
            selection_seeds=[99],
        )
        decisions.setdefault(
            row["state_id"],
            {
                "selected": 2,
                "abstained": False,
                "candidate_ids": ["k0", "k1", "k2"],
                "utility_mean": [-1, 0, 1],
                "risk_score": [-1, 0, 1],
                "h1": [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
            },
        )
    report = independent_selection_report(rows, decisions, (1 / 3,) * 3)
    assert report["trajectories"] == 3 and report["states"] == 6
    comparison = report["comparisons"][report["primary_comparison"]]
    assert comparison["units"] == 3 and comparison["mean"] > 0
    changed = copy.deepcopy(rows)
    changed[0]["selection_items"] = ["b"]
    with pytest.raises(ValueError, match="disjoint"):
        independent_selection_report(changed, decisions, (1 / 3,) * 3)
    changed = copy.deepcopy(rows)
    changed[0]["selection_seeds"] = [11]
    with pytest.raises(ValueError, match="seeds overlap"):
        independent_selection_report(changed, decisions, (1 / 3,) * 3)


def test_online_abstention_executes_no_full_update(tmp_path, monkeypatch):
    from third_eye.evaluation.runner import Consequences
    from third_eye.experiments.online import run_online
    from types import SimpleNamespace

    labeler = setup_labeler(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "third_eye.experiments.online.evaluate",
        lambda *args: Consequences(0.4, 0.2, 0.6),
    )
    fake = SimpleNamespace(
        metadata={
            "kind": "expected",
            "horizon": 2,
            "utility_weights": [1 / 3] * 3,
            "weights_sha256": "frozen",
            "selection_state_ids": [],
            "selection_trajectory_ids": [],
        },
        decision=lambda views: {"selected": None, "abstained": True},
    )
    run_online(
        labeler.backend,
        labeler.config,
        labeler.splits,
        labeler.verifier,
        tmp_path,
        labeler.manifest_hash,
        "expected",
        fake,
    )
    assert labeler.backend.starts == [] and labeler.backend.state == ()
    row = json.loads((tmp_path / "ledger.jsonl").read_text())
    assert row["candidate_id"] is None and row["training"]["optimizer_steps"] == 0
    assert row["forecast_decision"]["abstained"]


def test_repeated_terminal_collection(tmp_path, monkeypatch):
    from test_a1 import amended
    import third_eye.experiments.labeling as module

    labeler = setup_labeler(tmp_path, monkeypatch)
    labeler.config = amended(labeler.config)
    labeler.backend.config = labeler.config
    labeler.continuation_seed_offsets = (50_000_000, 100_000_000)
    original = module.collect_corrections
    monkeypatch.setattr(
        module,
        "collect_corrections",
        lambda b, *a: ([], {}) if b.state else original(b, *a),
    )
    monkeypatch.setattr(
        "third_eye.experiments.continuations.collect_corrections",
        lambda b, *a: ([], {}),
    )
    rows = labeler.label_state(0)
    assert len(labeler.backend.starts) == 3
    mean, covariance, _ = repeated_targets(rows)
    np.testing.assert_allclose(mean, np.stack([labels(r, 1) for r in rows]))
    np.testing.assert_allclose(covariance, 0, atol=1e-30)


def test_synthetic_positive_control_can_detect_expected_selection_signal(tmp_path):
    """Power/sanity control only: this never counts as empirical advantage."""
    torch.set_num_threads(1)
    parts = split_records(repeated_records())
    fit = train_forecaster(
        parts,
        tmp_path / "positive",
        kind="expected",
        hidden=16,
        epochs=80,
        patience=15,
        learning_rate=0.005,
    )
    metrics = json.loads((tmp_path / "positive/metrics.json").read_text())["test"]
    assert metrics["ranking"]["spearman"] > 0.8
    assert metrics["ranking"]["top1"] > metrics["ranking"]["chance_top1"]
    assert fit.metadata["epochs_completed"] <= 80


def test_scoring_draws_do_not_include_placeholder_or_commit_future(
    tmp_path, monkeypatch
):
    from third_eye.experiments.continuations import collect_scoring_replicates

    labeler = setup_labeler(tmp_path / "native", monkeypatch)
    row = labeler.label_state(0)[0]
    saved = labeler.backend.state_hash()
    offsets = [150_000_000, 200_000_000, 250_000_000]
    seeds = [row["continuation"]["seed"] + o for o in offsets]
    observations = collect_scoring_replicates(labeler, row, tmp_path / "scoring", seeds)
    assert [r["seed"] for r in observations] == seeds
    assert labeler.backend.state_hash() == saved
    scored = dict(row, continuation_replicates=observations)
    repeated_targets(
        [scored, *[dict(scored, candidate_id=f"other{i}") for i in range(2)]]
    )


def test_split_fractions_remain_trajectory_disjoint_and_nonempty():
    parts = split_records(
        repeated_records(20), validation_fraction=0.15, test_fraction=0.15
    )
    assert len({r["trajectory_id"] for r in parts["train"]}) == 14
    assert len({r["trajectory_id"] for r in parts["validation"]}) == 3
    assert len({r["trajectory_id"] for r in parts["test"]}) == 3
    with pytest.raises(ValueError, match="fractions"):
        split_records(repeated_records(3), validation_fraction=0.45, test_fraction=0.45)


def test_matched_reference_preserves_native_labels_and_is_not_an_input():
    from third_eye.forecasting.continuation import reference_h1

    rows = repeated_records(3)
    original = copy.deepcopy(rows)
    inputs = candidate_inputs(rows)
    for row in rows:
        reference = {h: 0.04 for h in ("target", "ood", "retention")}
        row["continuation_reference"] = {
            "schema_version": 1,
            "source": "matched_control_reevaluation",
            **{
                k: row[k]
                for k in (
                    "candidate_adapter_hash",
                    "parent_adapter_hash",
                    "manifest_hash",
                )
            },
            "h1": reference,
        }
        for draw in row["continuation_replicates"]:
            draw.update(
                continuation_available=0,
                optimizer_steps=0,
                continuation_adapter_hash=row["candidate_adapter_hash"],
                h2=reference,
            )
    mean, cov, _ = repeated_targets(rows)
    np.testing.assert_allclose(mean, 0.04)
    np.testing.assert_allclose(cov, 0, atol=1e-30)
    np.testing.assert_array_equal(inputs, candidate_inputs(rows))
    assert [r["labels"] for r in rows] == [r["labels"] for r in original]
    np.testing.assert_allclose(reference_h1(rows[0]), [0.04] * 3)
    rows[0]["continuation_reference"]["candidate_adapter_hash"] = "different"
    with pytest.raises(ValueError, match="reference checkpoint"):
        repeated_targets(rows)
