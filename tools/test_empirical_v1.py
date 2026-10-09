"""Synthetic controls for supplemental selection and inference contracts."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import empirical_cached_v1 as cached
import empirical_common_v1 as common

spec = importlib.util.spec_from_file_location(
    "empirical_fixture",
    Path(__file__).resolve().parents[1] / "tests/test_forecasting.py",
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_selection_ties_random_and_holding_negative_updates():
    rows = fixture.records(6)
    for g in cached.analysis.groups(rows):
        for j, i in enumerate(g):
            rows[i]["labels"]["h1"] = {h: float(j * 0.1) for h in cached.analysis.HEADS}
            rows[i]["labels"]["h2"] = {
                h: float(-j * 0.1 - 0.1) for h in cached.analysis.HEADS
            }
    r = cached.selection(rows)
    assert np.isclose(r["policies"]["random"]["future_utility"]["state_mean"], -0.2)
    assert np.isclose(
        r["policies"]["immediate_oracle"]["oracle_regret"]["state_mean"], 0.2
    )
    assert r["policies"]["hold_parameters"]["future_utility"]["state_mean"] == 0
    assert r["immediate_oracle_minus_random"]["state_mean"] < 0


def test_positive_control_preserves_real_labels_and_recovers_signal():
    rows = fixture.records(9)
    for r in rows:
        r["analysis_stream"] = "synthetic/math"
    train = cached.analysis.subset(rows, 6)
    val = [r for r in rows if r not in train]
    before = copy.deepcopy(rows)
    result = cached.positive_controls(train, val)
    assert rows == before and len(result) == 9
    assert all(
        r["metrics"]["spearman"]["state_mean"] > 0.99
        for r in result
        if r["injected_noise_sd"] == 0
    )


def test_noise_sample_is_metadata_only_distinct_trajectories_and_excludes_test():
    rows = []
    items = []
    for stream in ["a/math", "a/code", "b/math", "b/code"]:
        for t in range(4):
            tid = stream + str(t)
            items.append(
                {
                    "trajectory_id": tid,
                    "index": len(items),
                    "source": "/scratch/labels/"
                    + stream
                    + "/seed"
                    + str(t)
                    + "/random",
                }
            )
            for gen in [0, 2, 4]:
                for k in range(3):
                    rows.append(
                        {
                            "trajectory_id": tid,
                            "state_id": tid + str(gen),
                            "generation": gen,
                            "candidate_id": str(k),
                            "labels": {"future": 999},
                        }
                    )
    parts = {"train": rows, "validation": [], "test": []}
    a = common.select_noise({"trajectories": items}, parts)
    changed = copy.deepcopy(parts)
    for r in changed["train"]:
        r["labels"] = {"future": -999}
    b = common.select_noise({"trajectories": items}, changed)
    assert [(r["state_id"], r["trajectory_id"]) for r in a] == [
        (r["state_id"], r["trajectory_id"]) for r in b
    ]
    assert len(a) == len({r["trajectory_id"] for r in a}) == 12
    assert {r["generation"] for r in a} == {0, 2, 4}
    with pytest.raises(RuntimeError):
        common.select_noise(
            {"trajectories": items}, {"train": [], "validation": [], "test": rows}
        )
