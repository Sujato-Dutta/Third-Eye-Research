"""Shared evaluation noise and fixed actions must survive confirmation reporting."""

import copy

import numpy as np
import pytest

from third_eye.experiments.confirmation_statistics import bootstrap, values, summarize
from third_eye.experiments.confirmation import DECLARED


def cohort():
    result = []
    for family in ("llama", "qwen"):
        for task in ("code", "math"):
            for seed in range(2):
                parent = np.array([0, 1] * 4, float)
                future = np.zeros(8) if family == "llama" else np.ones(8)
                roles = ("target_test", "ood_test", "retention_test")
                artifacts = {"parent": {role: parent for role in roles}}
                for k in range(3):
                    artifacts[f"m1_{k}"] = {role: parent for role in roles}
                    for d in range(3):
                        artifacts[f"m2_{k}_{d}"] = {role: future for role in roles}
                result.append(
                    {
                        "stream": family + "/" + task,
                        "seed": seed,
                        "artifacts": artifacts,
                        "ids": {
                            role: [(role + str(i), str(i)) for i in range(8)]
                            for role in roles
                        },
                        "actions": {
                            "hold": [0, 0, 0],
                            "forced": [1, 0, 0],
                            "replay": [1, 0, 0],
                            "family_admission": [1, 0, 0]
                            if family == "qwen"
                            else [0, 0, 0],
                        },
                        "terminal_draws": 0,
                    }
                )
    return result


def test_identical_policies_share_all_parent_and_item_resamples():
    for mode in ("parent", "item", "joint"):
        result = bootstrap(cohort(), draws=100, mode=mode)
        np.testing.assert_array_equal(result["boot"][:, 1], result["boot"][:, 2])
        np.testing.assert_array_equal(result["boot"][:, 0], np.zeros(100))
        assert abs(result["point"][-1] - 1) < 1e-12


def test_partial_shared_pool_is_rejected_before_misleading_interval():
    states = cohort()
    states[0]["ids"]["retention_test"][0] = ("different", "different")
    with pytest.raises(ValueError, match="Partially"):
        bootstrap(states, draws=10)


def test_future_outcomes_affect_scoring_but_cannot_change_saved_actions():
    state = cohort()[0]
    saved = copy.deepcopy(state["actions"])
    before = values(state, "target_test")
    for k in range(3):
        for d in range(3):
            state["artifacts"][f"m2_{k}_{d}"]["target_test"] = np.ones(8)
    after = values(state, "target_test")
    assert state["actions"] == saved
    assert not np.array_equal(before[1], after[1])


def test_complete_synthetic_results_package_writes_all_declared_outputs(
    tmp_path, monkeypatch
):
    import json

    monkeypatch.setitem(DECLARED, "bootstrap_draws", 40)
    summarize(
        cohort(),
        tmp_path,
        {"source_tree_sha256": "fixture", "automatic_expansion": False},
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["forecaster_fits"] == 0
    assert report["model_difference_replicated"]
    assert len(report["secondary"]) == 6
    assert report["decision"] == "EXPERIMENTS_COMPLETE_FREEZE_EVIDENCE_AND_WRITE"
    assert all(
        (tmp_path / name).is_file()
        for name in [
            "policies.csv",
            "findings.md",
            "policy_utility.pdf",
            "policy_utility.png",
        ]
    )
