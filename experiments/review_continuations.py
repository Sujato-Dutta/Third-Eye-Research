"""Audit existing matched controls using the core repeated-target contracts.

These parents have already been inspected. Any fit made here is exploratory
development and cannot be presented as fresh confirmation.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from third_eye.cluster import require_compute_allocation
from third_eye.forecasting.continuation import repeated_targets, reliability_audit
from third_eye.forecasting.encoding import HEADS, labels
from third_eye.io import file_digest, publish_jsonl_batch, write_json
from third_eye.provenance import source_inventory


def attach_controls(task, controls):
    if controls.get("status") != "complete" or controls["state_id"] != task["state_id"]:
        raise ValueError("Native task and completed controls do not match")
    if controls["trajectory_id"] != task["trajectory_id"]:
        raise ValueError("Control trajectory identity differs")
    native = {r["candidate_id"]: r for r in task["selected_records"]}
    if set(native) != {c["candidate_id"] for c in controls["candidates"]}:
        raise ValueError("Incomplete matched candidate coverage")
    parent = controls["parent"]
    rows = []
    for candidate in controls["candidates"]:
        row = copy.deepcopy(native[candidate["candidate_id"]])
        conditions = candidate["conditions"]
        if (
            row["parent_adapter_hash"] != parent["adapter_hash"]
            or row["candidate_adapter_hash"]
            != conditions["original_m1"]["adapter_hash"]
        ):
            raise ValueError("Control M1/parent checkpoint differs")
        if row["batch_hash"] != candidate["batch_hash"]:
            raise ValueError("Candidate batch changed")
        h1 = np.asarray(
            [
                conditions["original_m1"]["metrics"][h] - parent["metrics"][h]
                for h in HEADS
            ]
        )
        row["continuation_reference"] = {
            "schema_version": 1,
            "source": "matched_control_reevaluation",
            "candidate_adapter_hash": row["candidate_adapter_hash"],
            "parent_adapter_hash": row["parent_adapter_hash"],
            "manifest_hash": row["manifest_hash"],
            "h1": dict(zip(HEADS, h1.tolist())),
            "native_h1_difference": (h1 - labels(row, 1)).tolist(),
        }
        names = ["original_m2"] + sorted(
            n for n in conditions if n.startswith("fresh_m2_")
        )
        if len(names) < 3:
            raise ValueError("Insufficient fresh continuation controls")
        observations = []
        for name in names:
            value = conditions[name]
            # Confirm all condition outcomes refer to the same scoring items.
            for role in ("target_dev", "ood_dev", "retention_dev"):
                base_ids = [
                    (i["id"], i["prompt_sha256"]) for i in parent["items"][role]
                ]
                if [
                    (i["id"], i["prompt_sha256"]) for i in value["items"][role]
                ] != base_ids:
                    raise ValueError("Control item identity/order differs")
                if any(i["correct"] not in (0, 1) for i in value["items"][role]):
                    raise ValueError("Invalid control verifier outcome")
            original = name == "original_m2"
            seed = row["continuation"]["seed"] if original else value["seed"]
            available = (
                row["continuation_available"]
                if original
                else value["continuation_available"]
            )
            training = row["runtime"]["continuation"] if original else value["training"]
            observations.append(
                {
                    "seed": seed,
                    "candidate_adapter_hash": row["candidate_adapter_hash"],
                    "continuation_adapter_hash": value["adapter_hash"],
                    "evaluation_manifest_hash": row["manifest_hash"],
                    "h2": {
                        h: value["metrics"][h] - parent["metrics"][h] for h in HEADS
                    },
                    "continuation_available": available,
                    "optimizer_steps": training["optimizer_steps"],
                    "reserved_optimizer_steps": 50,
                }
            )
        if (
            observations[0]["continuation_adapter_hash"]
            != row["continuation_adapter_hash"]
        ):
            raise ValueError(
                "Original H2 control checkpoint differs from native observation"
            )
        row["continuation_reference"]["native_h2_difference"] = (
            np.asarray([observations[0]["h2"][h] for h in HEADS]) - labels(row, 2)
        ).tolist()
        row["continuation_replicates"] = observations
        rows.append(row)
    repeated_targets(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-inputs", nargs="+", required=True)
    parser.add_argument("--controls", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exploratory-fit", action="store_true")
    args = parser.parse_args()
    require_compute_allocation()
    if len(args.native_inputs) != len(args.controls):
        parser.error("Pair each native input with its completed control")
    output = Path(args.output)
    if output.exists():
        parser.error("Continuation reviews are immutable")
    rows = []
    for source, control in zip(args.native_inputs, args.controls):
        rows.extend(
            attach_controls(
                json.loads(Path(source).read_text()),
                json.loads(Path(control).read_text()),
            )
        )
    output.mkdir(parents=True)
    publish_jsonl_batch(output / "repeated_labels.jsonl", rows)
    audit = reliability_audit(rows)
    audit.update(
        {
            "evidence_status": "Previously inspected parents; exploratory development only",
            "inputs": {p: file_digest(p) for p in args.native_inputs + args.controls},
            **source_inventory(),
        }
    )
    write_json(output / "audit.json", audit)
    if args.exploratory_fit:
        import torch
        from third_eye.forecasting.dataset import split_records
        from third_eye.forecasting.training import train_forecaster

        torch.set_num_threads(4)
        train_forecaster(
            split_records(rows),
            output / "forecaster",
            kind="expected",
            hidden=16,
            epochs=100,
            patience=15,
        )


if __name__ == "__main__":
    main()
