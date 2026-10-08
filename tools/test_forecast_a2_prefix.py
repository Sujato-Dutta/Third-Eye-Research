"""Published-prefix forecasting preserves complete states and frozen splits."""

from pathlib import Path

import pytest

from forecast_a2_prefix_20261008 import assignments, published_rows, remap


def state(trajectory, generation):
    return [
        {
            "trajectory_id": trajectory,
            "state_id": f"{trajectory}:{generation}",
            "candidate_id": str(k),
            "model": "fixed",
        }
        for k in range(3)
    ]


def test_exclude_unpublished_states_and_reject_incomplete_k():
    rows = state("a", 0)
    ledger = [{"status": "complete", "state_id": "a:0", "candidates": 3}]
    assert published_rows(rows + state("a", 1), ledger) == rows
    with pytest.raises(AssertionError):
        published_rows(rows[:2], ledger)
    with pytest.raises(AssertionError):
        published_rows([rows[0]] * 3, ledger)


def test_final_generation_does_not_move_trajectories_between_partitions():
    prefix = [r for t in range(36) for g in range(4) for r in state(str(t), g)]
    complete = [r for t in range(36) for g in range(5) for r in state(str(t), g)]
    assert assignments(prefix) == assignments(complete)
    parts = assignments(prefix)
    assert [len(parts[p]) for p in ["train", "validation", "test"]] == [20, 8, 8]
    assert not set(parts["train"]) & set(parts["validation"])


def test_commands_keep_frozen_settings_and_separate_outputs():
    before = [
        "train.py",
        "--labels",
        "old.jsonl",
        "--output",
        "/original/forecasters/direct",
        "--device",
        "cpu",
        "--split-seed",
        "42",
        "--seed",
        "43",
        "--horizon",
        "2",
    ]
    after = remap(before, ["snapshot.jsonl"], Path("/original"), Path("/provisional"))
    assert after == [
        "train.py",
        "--labels",
        "snapshot.jsonl",
        "--output",
        "/provisional/forecasters/direct",
        "--device",
        "cpu",
        "--split-seed",
        "42",
        "--seed",
        "43",
        "--horizon",
        "2",
    ]
    assert before[2] == "old.jsonl"


def test_launcher_is_cpu_and_uses_unix_line_endings():
    data = (Path(__file__).parent / "prefix_forecast_a2_20261008.slurm").read_bytes()
    assert b"\r" not in data and b"--partition=gg" in data
    assert b'CUDA_VISIBLE_DEVICES=""' in data
