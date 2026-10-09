# ruff: noqa: E402
"""Compute-node audit of actual backbone files and retained recovery checkpoints."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_common_v1 import BASE, checked, prepared
from third_eye.config import Config
from third_eye.io import digest, file_digest, write_json
from third_eye.training.sources import resolve_source


def main():
    checked()
    tasks = prepared()["noise_tasks"]
    out = BASE / "input_audit_v1"
    out.mkdir(exist_ok=False)
    proofs = {}
    metadata = []
    for task in tasks:
        cfg = Config.load(task["config"])
        run = json.loads((Path(task["source"]) / "run.json").read_text())
        manifest = json.loads(Path(task["manifest"]).read_text())
        if (
            cfg.to_dict() != run["config"]
            or cfg.fingerprint != run["config_hash"]
            or digest(manifest) != run["manifest_hash"]
        ):
            raise RuntimeError(
                "Current configuration/manifest differs from original run"
            )
        if any(
            r["config_hash"] != cfg.fingerprint
            or r["manifest_hash"] != digest(manifest)
            for r in task["selected_records"]
        ):
            raise RuntimeError("Original label input fingerprint differs")
        _, _, proof = resolve_source(cfg.model)
        if proof != run["model_source"]:
            raise RuntimeError("Original model source proof changed")
        if cfg.model.name not in proofs:
            hashes = {
                name: file_digest(Path(proof["runtime_directory"]) / name)
                for name in proof["verified_weight_sha256"]
            }
            write_json(
                out / (cfg.model.name.replace("/", "_") + "_weights.json"),
                dict(actual=hashes, expected=proof["verified_weight_sha256"]),
            )
            if hashes != proof["verified_weight_sha256"]:
                raise RuntimeError("Actual backbone weights changed")
            proofs[cfg.model.name] = hashes
        metadata.append(
            dict(
                index=task["index"],
                config_exact=True,
                manifest_exact=True,
                source_proof_exact=True,
            )
        )
    recovery = []
    rec = ROOT / "runs/a2/recovery_20261008_v3"
    for index in [20, 22, 24]:
        task = json.loads((rec / f"task_{index}.json").read_text())
        for branch in task["branches"]:
            if branch["complete"]:
                continue
            directory = Path(task["partial_state"]) / branch["candidate_id"]
            available = (directory / "t1/state.json").exists()
            item = dict(
                index=index, candidate_id=branch["candidate_id"], retained_t1=available
            )
            if available:
                state = json.loads((directory / "t1/state.json").read_text())
                log = [
                    json.loads(line)
                    for line in (directory / "training_t1.jsonl")
                    .read_text()
                    .splitlines()
                ]
                item.update(
                    checkpoint_files_exact=all(
                        file_digest(directory / "t1" / p) == h
                        for p, h in state["weights"].items()
                    ),
                    fifty_complete_steps=[r["step"] for r in log] == list(range(1, 51)),
                    adapter_hash=state["adapter_hash"],
                )
            recovery.append(item)
    write_json(
        out / "completed.json",
        dict(
            status="actual_input_audit_completed",
            base_weights=proofs,
            original_inputs=metadata,
            incomplete_recovery_branches=recovery,
            original_data_changed=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="actual_input_audit_completed",
                incomplete_recovery_branches=recovery,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
