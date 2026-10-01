"""Fit a compact pre-commit forecaster on trajectory-disjoint cached labels."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.forecasting.dataset import read_records, split_records
from third_eye.forecasting.training import train_forecaster
from third_eye.cluster import require_gpu_allocation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--kind",
        choices=("one_step", "matched_h1", "direct", "dynamics"),
        default="direct",
    )
    p.add_argument("--horizon", type=int, choices=(1, 2), default=2)
    p.add_argument("--scalar", action="store_true")
    p.add_argument(
        "--ablate",
        nargs="*",
        choices=("gradient", "probe", "history", "diversity", "retention"),
        default=[],
    )
    p.add_argument("--held-out-model", action="append", default=[])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--utility-weights", nargs=3, type=float, default=(1 / 3, 1 / 3, 1 / 3)
    )
    a = p.parse_args()
    require_gpu_allocation(a.device)
    records = read_records(a.labels)
    parts = split_records(records, a.split_seed, a.held_out_model)
    train_forecaster(
        parts,
        a.output,
        a.kind,
        a.horizon,
        a.utility_weights,
        a.ablate,
        a.seed,
        a.epochs,
        a.patience,
        a.hidden,
        a.learning_rate,
        a.scalar,
        a.device,
    )
    print(f"Saved {a.kind} H={a.horizon} forecaster to {a.output}")


if __name__ == "__main__":
    main()
