"""Produce phenomenon/forecast reports and measured Gate 1/2 artifacts."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.forecasting.dataset import read_records
from third_eye.io import file_digest, write_json
from third_eye.statistics.report import phenomenon, gate1, gate2, forecast_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--forecaster")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        p.error("Reports are immutable; use a new output directory")
    records = read_records(a.labels)
    report = phenomenon(records)
    write_json(out / "phenomenon.json", report)
    write_json(
        out / "gate1.json",
        {
            **gate1(report),
            "evidence_hashes": {str(path): file_digest(path) for path in a.labels},
        },
    )
    if a.forecaster:
        from third_eye.forecasting.training import Forecaster

        f = Forecaster.load(a.forecaster)
        chosen_states = set(f.metadata["selection_state_ids"])
        chosen_trajectories = set(f.metadata["selection_trajectory_ids"])
        test = [
            r
            for r in records
            if r["state_id"] not in chosen_states
            and r["trajectory_id"] not in chosen_trajectories
        ]
        if test:
            write_json(out / "forecast_test.json", forecast_report(test, f))
        metrics = json.loads(
            (Path(a.forecaster) / "metrics.json").read_text(encoding="utf-8")
        )
        write_json(
            out / "gate2.json",
            {
                **gate2(
                    metrics["validation"], f.metadata["kind"], f.metadata["horizon"]
                ),
                "forecaster_weights_sha256": f.metadata["weights_sha256"],
                "validation_split_hash": f.metadata["split_hashes"]["validation"],
            },
        )
    print(f"Saved evidence reports to {out}")


if __name__ == "__main__":
    main()
