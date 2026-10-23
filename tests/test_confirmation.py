"""Fresh confirmation decisions must not inspect future or scoring outcomes."""

from pathlib import Path

import numpy as np

from third_eye.experiments.confirmation import actions, checkpoints, primary_power


def test_frozen_rules_use_development_only_and_never_reselect_candidate():
    immediate = np.array([[0.02, 0.02, 0], [0, 0, 0], [-0.1, 0, 0]])
    result = actions("qwen/math", immediate)
    assert set(result) == {"hold", "forced", "replay", "family_admission"}
    assert result["forced"] == [1, 0, 0]
    assert all(mass[1:] == [0, 0] for mass in result.values())
    assert actions("llama/math", immediate)["family_admission"] == [0, 0, 0]


def test_checkpoints_include_all_thirteen_and_use_saved_terminal_path():
    rows = [
        {
            "checkpoints": {
                "t1": f"/state/candidate{i}/t1",
                "t2": f"/state/candidate{i}/t2",
            },
            "candidate_adapter_hash": f"m1{i}",
            "parent_adapter_hash": "parent",
            "continuation_replicates": [
                {"seed": d + 100, "continuation_adapter_hash": f"m2{i}{d}"}
                for d in range(3)
            ],
        }
        for i in range(3)
    ]
    output = checkpoints(rows)
    assert len(output) == 13 and len({name for name, _, _ in output}) == 13
    assert output[0] == ("parent", Path("/state/parent"), "parent")
    assert (
        "m2_0_1",
        Path("/state/candidate0/continuation_replicates/101/t2"),
        "m201",
    ) in output


def test_single_primary_power_does_not_require_pooled_universal_harm():
    rng = np.random.default_rng(42)
    point = np.array([0.01, 0, 0, 0, 0, 0, 0.10])
    boot = point + rng.normal(0, 0.001, (1000, 7))
    assert primary_power(point, boot)["conditional_planned_power"] > 0.99
