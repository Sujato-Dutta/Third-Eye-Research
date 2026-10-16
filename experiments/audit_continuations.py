"""Audit repeated continuation signal on a Slurm CPU or GPU compute node."""

import argparse
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.forecasting.continuation import reliability_audit
from third_eye.forecasting.dataset import read_records
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if "login" in socket.gethostname().lower() or not os.environ.get("SLURM_JOB_ID"):
        parser.error("Scientific analysis requires a Slurm compute allocation")
    if Path(args.output).exists():
        parser.error("Audit outputs are immutable")
    result = reliability_audit(read_records(args.labels))
    result["inputs"] = {p: file_digest(p) for p in args.labels}
    result.update(source_inventory())
    write_json(args.output, result)


if __name__ == "__main__":
    main()
