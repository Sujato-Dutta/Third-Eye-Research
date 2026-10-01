"""Freeze measured pilot settings without overwriting the pilot config."""

import argparse
from dataclasses import replace
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.config import Config
from third_eye.io import file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Chosen measured pilot config")
    parser.add_argument(
        "--pilot-run",
        required=True,
        help="Completed run directory using the chosen config",
    )
    parser.add_argument("--output", required=True, help="New frozen config filename")
    parser.add_argument(
        "--depth",
        type=int,
        choices=range(1, 6),
        help="Frozen recursive depth; per-update budgets stay measured",
    )
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Frozen configs are immutable; choose a new output path")
    cfg = Config.load(args.config)
    run_dir = Path(args.pilot_run)
    run = json.loads((run_dir / "run.json").read_text())
    completed = json.loads((run_dir / "completed.json").read_text())
    if (
        run["config_hash"] != cfg.fingerprint
        or completed["status"] != "complete"
        or not completed["accepted"]
    ):
        parser.error("Chosen settings must correspond to a completed measured pilot")
    if not (run_dir / "meta_labels.jsonl").exists():
        parser.error("Pilot did not produce labels")
    revision = run["resolved_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("Pilot must record the immutable Hugging Face model commit")
    frozen = replace(
        cfg,
        model=replace(cfg.model, revision=revision),
        protocol=replace(
            cfg.protocol, status="frozen", depth=args.depth or cfg.protocol.depth
        ),
    )
    write_json(args.output, frozen.to_dict())
    write_json(
        str(args.output) + ".freeze.json",
        {
            "pilot_run": str(run_dir.resolve()),
            "pilot_config_hash": cfg.fingerprint,
            "frozen_config_hash": frozen.fingerprint,
            "pilot_labels_sha256": file_digest(run_dir / "meta_labels.jsonl"),
            "resolved_revision": revision,
            "frozen_depth": frozen.protocol.depth,
        },
    )
    print(f"Frozen config: {args.output}")


if __name__ == "__main__":
    main()
