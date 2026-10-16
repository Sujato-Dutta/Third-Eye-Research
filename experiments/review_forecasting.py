"""Run a bounded development forecasting comparison on inspected pilot data."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.cluster import require_compute_allocation
from third_eye.forecasting.dataset import read_records
from third_eye.forecasting.review import run_review
from third_eye.forecasting.training import Forecaster
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory
from forecasting_pilots import inspect_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-plan", required=True)
    p.add_argument("--inspection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frozen-forecaster", required=True)
    p.add_argument("--frozen-weights-sha256", required=True)
    args = p.parse_args()
    require_compute_allocation()
    plan = json.loads(Path(args.pilot_plan).read_text())
    inspection = json.loads(Path(args.inspection).read_text())
    if inspection.get("passed") is not True or len(inspection["tasks"]) != 8:
        raise ValueError("Eight structurally inspected pilots are required")
    paths = [str(Path(t["output"]) / "meta_labels.jsonl") for t in plan["tasks"]]
    records = read_records(paths)
    streams = {}
    for task in plan["tasks"]:
        inspect_task(
            task, plan["continuation_seed_offsets"], plan["source_tree_sha256"]
        )
        for row in read_records([str(Path(task["output"]) / "meta_labels.jsonl")]):
            streams[row["state_id"]] = task["stream"]
    output = Path(args.output)
    frozen = Forecaster.load(args.frozen_forecaster)
    if frozen.metadata["weights_sha256"] != args.frozen_weights_sha256:
        raise ValueError("Frozen checkpoint differs from the predeclared weights")
    run_review(records, output, streams, frozen_model=frozen)
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "development_only": True,
            "inputs": {
                path: file_digest(path)
                for path in paths + [args.pilot_plan, args.inspection]
            },
            **source_inventory(),
        },
    )
    print(
        "Completed development forecasting review; no automatic expansion or Gate 2 decision"
    )


if __name__ == "__main__":
    main()
