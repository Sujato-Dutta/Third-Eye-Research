"""Prepare immutable benchmark suites; run inside a scheduled CPU job."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.data.benchmarks import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("math", "code"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--dev-size", type=int, default=64)
    p.add_argument("--retention-size", type=int, default=256)
    a = p.parse_args()
    if min(a.train_size, a.dev_size, a.retention_size) < 1:
        p.error("Split sizes must be positive")
    print(prepare(a.output, a.task, a.seed, a.train_size, a.dev_size, a.retention_size))


if __name__ == "__main__":
    main()
