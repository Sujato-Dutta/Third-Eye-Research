"""Test resource handoff invariants without touching scientific artifacts."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "parallel", Path(__file__).with_name("a1_parallel_resume.py")
)
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


def grid():
    return [
        {"stream": s, "seed": seed}
        for s in sorted(parallel.STREAMS)
        for seed in parallel.SEEDS
    ]


def test_adopts_submitted_tasks_without_repeating_them():
    submissions = [{"indices": [i, i + 5, i + 10, i + 15]} for i in range(3)]
    assert parallel.pending_indices(grid(), submissions) == [3, 4, 8, 9, 13, 14, 18, 19]


def test_rejects_duplicate_submissions():
    with pytest.raises(ValueError, match="duplicate"):
        parallel.pending_indices(grid(), [{"indices": [0]}, {"indices": [0]}])


def test_rejects_modified_grid():
    tasks = grid()
    tasks[-1]["seed"] = 6042
    with pytest.raises(ValueError, match="unchanged"):
        parallel.pending_indices(tasks, [])


def test_counts_adopted_reservations_and_new_jobs():
    submissions = [
        {"indices": [0], "reserved_hours": 8, "billed_hours": 3},
        {"indices": [1], "reserved_hours": 8},
    ]
    assert parallel.reservation(50, submissions, [2, 3], 20, 100) == 24
    with pytest.raises(RuntimeError, match="resource envelope"):
        parallel.reservation(50, submissions, [2, 3], 20, 73)


def test_rejects_excess_parallelism():
    with pytest.raises(ValueError):
        parallel.reservation(0, [], [], 21, 100)


def test_array_success_requires_all_expected_tasks():
    ok = ["123_1", "COMPLETED", "100", "0:0"]
    assert parallel.terminal_success([ok, ok], 2)
    assert not parallel.terminal_success([ok], 2)
    assert not parallel.terminal_success([ok, ["123_2", "FAILED", "2", "1:0"]], 2)
