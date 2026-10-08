"""The single diagnostic changes patience, keeping matched settings intact."""

from pathlib import Path

from forecast_a2_patience_diag_20261008 import diagnostic_command, BASE, OUT


def test_only_forecaster_patience_changes_within_existing_epoch_cap():
    before = [
        "fit.py",
        "--labels",
        "old.jsonl",
        "--output",
        (BASE / "study/forecasters/direct_seed42").as_posix(),
        "--kind",
        "direct",
        "--horizon",
        "2",
        "--seed",
        "42",
        "--split-seed",
        "42",
        "--device",
        "cpu",
        "--epochs",
        "200",
        "--patience",
        "20",
    ]
    after = diagnostic_command(before, ["frozen_snapshot.jsonl"])
    assert after[after.index("--patience") + 1] == "200"
    assert after[after.index("--epochs") + 1] == "200"
    assert (
        after[after.index("--output") + 1]
        == (OUT / "study/forecasters/direct_seed42").as_posix()
    )
    for flag in ["--kind", "--horizon", "--seed", "--split-seed", "--device"]:
        assert after[after.index(flag) + 1] == before[before.index(flag) + 1]
    assert before[-1] == "20"


def test_launcher_is_short_compute_only_and_unix():
    data = (
        Path(__file__).parent / "forecast_a2_patience_diag_20261008.slurm"
    ).read_bytes()
    assert b"\r" not in data and b"--time=00:15:00" in data
    assert b'CUDA_VISIBLE_DEVICES=""' in data
