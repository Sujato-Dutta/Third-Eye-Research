"""Synthetic forecast contracts; fixtures are never empirical evidence."""

import copy
import json

import numpy as np
import pytest
import torch

from third_eye.forecasting.dataset import read_records, split_records
from third_eye.forecasting.encoding import encode
from third_eye.forecasting.training import train_forecaster, Forecaster
from third_eye.statistics.report import gate2


def records(trajectories=9):
    result = []
    for trajectory in range(trajectories):
        for generation in range(2):
            for candidate in range(3):
                x = (candidate - 1) * 0.1
                result.append(
                    {
                        "schema_version": 1,
                        "state_id": f"s{trajectory}-{generation}",
                        "trajectory_id": f"t{trajectory}",
                        "candidate_id": f"k{candidate}",
                        "generation": generation,
                        "model": "core",
                        "seed": trajectory,
                        "config_hash": "cfg",
                        "manifest_hash": "data",
                        "parent_adapter_hash": f"p{trajectory}-{generation}",
                        "candidate_size": 4,
                        "precommit": {
                            "state": {"target": 0.4, "ood": 0.2, "retention": 0.6},
                            "history": [],
                            "features": {
                                "pre_update_nll": candidate + 1,
                                "completion_words_mean": 10 + candidate,
                            },
                            "pool_statistics": {},
                            "generation": generation,
                        },
                        "labels": {
                            "h1": {"target": -x, "ood": 0.0, "retention": 0.0},
                            "h2": {"target": x, "ood": x / 2, "retention": -x / 3},
                            "utility_h1": -x / 3,
                            "utility_h2": (x + x / 2 - x / 3) / 3,
                        },
                    }
                )
    return result


def test_encoding_cannot_read_labels_or_identity_and_ablation_masks():
    original = records()[0]
    changed = copy.deepcopy(original)
    changed.update(
        labels={"h2": {"target": 99999}},
        model="different",
        state_id="different",
        seed=99,
        evaluation={"t2": 999},
        runtime={"loss": 999},
        checkpoints={"x": "private"},
    )
    for a, b in zip(encode(original), encode(changed)):
        np.testing.assert_array_equal(a, b)
    changed["precommit"]["history"] = [
        {
            "consequences": {"target": 1, "ood": 2, "retention": 3},
            "delta": {"target": 0.1, "ood": 0.2, "retention": 0.3},
        }
    ]
    assert encode(changed)[-1].sum() == 1
    assert encode(changed, ["history"])[-1].sum() == 0
    assert encode(changed, ["retention"])[0][2] == 0


def test_grouped_split_and_held_out_family():
    data = records()
    for row in data:
        if row["trajectory_id"] == "t8":
            row["model"] = "gemma"
    parts = split_records(data, held_out_models=["gemma"])
    assert all(r["model"] != "gemma" for r in parts["train"] + parts["validation"])
    for p, rows in parts.items():
        for other, other_rows in parts.items():
            if p != other:
                assert not {r["trajectory_id"] for r in rows} & {
                    r["trajectory_id"] for r in other_rows
                }
                assert not {r["state_id"] for r in rows} & {
                    r["state_id"] for r in other_rows
                }


def test_incomplete_states_rejected_and_merged_shared_states(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records()[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        read_records([path])
    data = records()
    # Two trajectories share a state: their entire trajectories stay together.
    for r in data:
        if r["trajectory_id"] == "t1" and r["generation"] == 0:
            r["state_id"] = "s0-0"
    parts = split_records(data)
    owners = {r["trajectory_id"]: p for p, rows in parts.items() for r in rows}
    assert owners["t0"] == owners["t1"]


@pytest.mark.parametrize(
    "kind,horizon", [("one_step", 1), ("matched_h1", 1), ("direct", 2), ("dynamics", 2)]
)
def test_train_roundtrip_and_inference_without_labels(tmp_path, kind, horizon):
    torch.set_num_threads(1)
    parts = split_records(records())
    model = train_forecaster(
        parts,
        tmp_path / kind,
        kind=kind,
        horizon=horizon,
        epochs=4,
        patience=3,
        hidden=16,
    )
    views = [
        {
            "precommit": r["precommit"],
            "state_id": r["state_id"],
            "candidate_id": r["candidate_id"],
        }
        for r in parts["test"]
    ]
    predictions = model.predict(views)
    assert predictions.shape == (len(views), 3)
    assert np.isfinite(predictions).all()
    loaded = Forecaster.load(tmp_path / kind)
    np.testing.assert_array_equal(predictions, loaded.predict(views))
    assert loaded.metadata["trainable_parameters"] < 1_000_000
    metrics = json.loads((tmp_path / kind / "metrics.json").read_text())
    assert metrics["test"]["future_h2"]["horizon"] == 2
    if kind == "dynamics":
        assert metrics["test"]["latent_rollout"]["h2_mse"] >= 0
    assert 0 <= loaded.select(views[:3]) < 3
    (tmp_path / kind / "weights.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        Forecaster.load(tmp_path / kind)


def test_normalizer_fit_only_on_training_and_leakage_rejected(tmp_path):
    torch.set_num_threads(1)
    parts = split_records(records())
    for r in parts["test"]:
        r["precommit"]["features"]["completion_words_mean"] = 999999
    f = train_forecaster(parts, tmp_path / "valid", epochs=1, hidden=8)
    assert max(f.metadata["mean"]) < 100
    parts["validation"] = parts["train"][:3]
    with pytest.raises(ValueError, match="leakage"):
        train_forecaster(parts, tmp_path / "leaked", epochs=1, hidden=8)


def test_gate2_cannot_pass_ties_or_one_step_forecaster():
    metrics = {
        "states": 100,
        "nonconstant_states": 0,
        "ranking": {"spearman": None, "top1": 1.0},
    }
    assert not gate2(metrics, "direct", 2)["passed"]
    metrics["nonconstant_states"] = 40
    metrics["ranking"]["spearman"] = 0.6
    assert gate2(metrics, "direct", 2)["passed"]
    assert not gate2(metrics, "one_step", 1)["passed"]
