"""Publish the preselected Gate 2 comparison before auxiliary fits finish."""

import json
from pathlib import Path

import pytest

import forecast_a2_prefix_fast_20261008 as fast


def test_priority_keeps_original_tasks_and_fixed_decision_seed():
    def task(kind, seed, extra=()):
        return {"commands": [["fit", "--kind", kind, "--seed", str(seed), *extra]]}

    tasks = [
        task("one_step", 42),
        task("direct", 43),
        task("direct", 42),
        task("direct", 42, ["--ablate", "gradient"]),
        task("matched_h1", 42),
    ]
    before = json.dumps(tasks)
    assert fast.priority_indices(tasks) == [2, 4]
    assert json.dumps(tasks) == before


def test_early_review_uses_fixed_validation_gate_and_matching_splits(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fast, "OUT", tmp_path)
    study = tmp_path / "study"
    metrics = {
        "states": 40,
        "nonconstant_states": 30,
        "ranking": {
            "spearman": 0.30,
            "top1": 0.45,
            "informative_top1": 0.45,
            "informative_chance_top1": 1 / 3,
        },
    }
    for kind, horizon in [("direct", 2), ("matched_h1", 1)]:
        p = study / "forecasters" / f"{kind}_seed42"
        p.mkdir(parents=True)
        (p / "metadata.json").write_text(
            json.dumps(
                {
                    "seed": 42,
                    "kind": kind,
                    "horizon": horizon,
                    "weights_sha256": kind,
                    "split_hashes": {"validation": "fixed"},
                    "utility_weights": [1 / 3] * 3,
                }
            )
        )
        (p / "metrics.json").write_text(
            json.dumps({"validation": {**metrics, "future_h2": metrics}})
        )
    fast.early_review(study, {"states": 177})
    report = json.loads((tmp_path / "early_gate2_review.json").read_text())
    assert report["gate2"]["passed"]
    assert report["fixed_decision_seed"] == 42 and not report["online_submitted"]
    p = study / "forecasters/matched_h1_seed42/metadata.json"
    wrong = json.loads(p.read_text())
    wrong["split_hashes"]["validation"] = "other"
    p.write_text(json.dumps(wrong))
    with pytest.raises(AssertionError):
        fast.early_review(study, {"states": 177})


def test_fast_launcher_is_bounded_and_uses_unix_line_endings():
    data = (
        Path(__file__).parent / "prefix_forecast_a2_fast_20261008.slurm"
    ).read_bytes()
    assert b"\r" not in data and b"--time=02:00:00" in data
    assert b'CUDA_VISIBLE_DEVICES=""' in data
