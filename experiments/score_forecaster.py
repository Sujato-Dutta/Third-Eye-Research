"""Independent-seed, disjoint-item H2 scoring for a frozen Third Eye selector.

Run once on fresh held-out trajectories after freezing the model. This scores
all K=3 saved M1s for matched baselines; it does not advance a trajectory.
"""

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.cluster import require_gpu_allocation, require_compute_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.evaluation.runner import evaluate
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.continuations import collect_scoring_replicates
from third_eye.forecasting.continuation import (
    HEADS,
    reference_h1,
    repeated_targets,
    state_groups,
)
from third_eye.forecasting.dataset import read_records
from third_eye.forecasting.scoring import independent_selection_report
from third_eye.forecasting.training import Forecaster
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--selection-manifest", required=True)
    p.add_argument("--scoring-manifest", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--forecaster", required=True)
    p.add_argument("--scoring-seed-offsets", nargs="+", type=int, required=True)
    p.add_argument(
        "--verifier",
        required=True,
        help="Trusted verifier factory with isolated code execution",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()
    config = Config.load(args.config)
    require_compute_allocation()
    require_gpu_allocation(config.model.device)
    if config.protocol.status != "frozen":
        p.error("Independent scoring requires a declared frozen protocol")
    output = Path(args.output)
    if output.exists():
        p.error("Independent scoring outputs are immutable")
    selection_manifest, selection = load_manifest(args.selection_manifest)
    scoring_manifest, scoring = load_manifest(args.scoring_manifest)
    if digest([asdict(e) for e in selection["train"]]) != digest(
        [asdict(e) for e in scoring["train"]]
    ):
        p.error("Scoring must retain exactly the declared correction-harvesting pool")
    selection_items = {
        e.prompt_hash for role, items in selection.items() for e in items
    }
    scoring_items = {
        e.prompt_hash
        for role, items in scoring.items()
        if role != "train"
        for e in items
    }
    if selection_items & scoring_items:
        p.error(
            "Scoring items must be disjoint from all selection/feature/training items"
        )
    offsets = args.scoring_seed_offsets
    if len(offsets) < 3 or len(set(offsets)) != len(offsets) or min(offsets) <= 0:
        p.error("Predeclare at least three distinct positive scoring seed offsets")
    rows = read_records(args.labels)
    forecaster = Forecaster.load(args.forecaster)
    if forecaster.metadata["kind"] != "expected":
        p.error("This scorer expects the core expected-value forecaster")
    if list(config.protocol.utility_weights) != forecaster.metadata["utility_weights"]:
        p.error("Scoring and forecaster utility weights differ")
    if any(
        r["config_hash"] != config.fingerprint
        or r["manifest_hash"] != digest(selection_manifest)
        for r in rows
    ):
        p.error("Checkpoint config/selection manifest does not match the native labels")
    for row in rows:
        if (
            row["trajectory_id"] in forecaster.metadata["selection_trajectory_ids"]
            or row["state_id"] in forecaster.metadata["selection_state_ids"]
        ):
            p.error("Scoring state/trajectory was used to develop the forecaster")
        native_seeds = {r["seed"] for r in row.get("continuation_replicates", [])}
        native_seeds.add(row["continuation"]["seed"])
        if native_seeds & {row["continuation"]["seed"] + o for o in offsets}:
            p.error("Scoring seeds overlap selection continuation seeds")
    from third_eye.training.hf_backend import HFBackend

    backend = HFBackend(config)
    verifier = VerifierRegistry.from_spec(args.verifier)
    output.mkdir(parents=True)
    write_json(
        output / "plan.json",
        {
            "forecaster_sha256": forecaster.metadata["weights_sha256"],
            "forecaster_metadata_hash": digest(forecaster.metadata),
            "selection_items": sorted(selection_items),
            "scoring_items": sorted(scoring_items),
            "scoring_seed_offsets": offsets,
            "inputs": {f: file_digest(f) for f in args.labels},
            "selection_manifest_hash": digest(selection_manifest),
            "scoring_manifest_hash": digest(scoring_manifest),
            **source_inventory(),
        },
    )
    measured, decisions = [], {}
    saved, saved_hash = backend.snapshot(), backend.state_hash()
    try:
        for indices in state_groups(rows):
            state_rows = [rows[i] for i in indices]
            views = [
                {k: r[k] for k in ("state_id", "candidate_id", "precommit")}
                for r in state_rows
            ]
            decisions[state_rows[0]["state_id"]] = forecaster.decision(views)
            write_json(
                output / state_rows[0]["state_id"] / "decision.json",
                decisions[state_rows[0]["state_id"]],
            )
            for row in state_rows:
                directory = output / row["state_id"] / row["candidate_id"]
                directory.mkdir(parents=True, exist_ok=True)
                checkpoint = Path(row["checkpoints"]["t1"])
                backend.load_checkpoint(checkpoint.parent.parent / "parent")
                if backend.state_hash() != row["parent_adapter_hash"]:
                    raise RuntimeError("Independent scorer restored a different parent")
                eval_seed = config.protocol.seed + row["generation"] * 1_000_000
                baseline = evaluate(
                    backend,
                    scoring,
                    verifier,
                    config.protocol.max_new_tokens,
                    eval_seed,
                    item_output=directory / "parent_evaluation.json",
                )
                backend.load_checkpoint(checkpoint)
                if backend.state_hash() != row["candidate_adapter_hash"]:
                    raise RuntimeError("Independent scorer restored a different M1")
                m1 = evaluate(
                    backend,
                    scoring,
                    verifier,
                    config.protocol.max_new_tokens,
                    eval_seed,
                    item_output=directory / "evaluation_t1.json",
                )
                scored = copy.deepcopy(row)
                # Historical matched references bind the selection manifest,
                # not these independent scoring items. Keep provenance separate.
                scored.pop("continuation_reference", None)
                scored["manifest_hash"] = digest(scoring_manifest)
                scored["labels"]["h1"] = m1.delta(baseline).to_dict()
                scored["evaluation"]["parent"] = baseline.to_dict()
                scored["evaluation_artifacts"] = {
                    "parent": str((directory / "parent_evaluation.json").resolve()),
                    "t1": str((directory / "evaluation_t1.json").resolve()),
                    "t2": str((directory / "evaluation_t1.json").resolve()),
                }
                labeler = SimpleNamespace(
                    backend=backend,
                    config=config,
                    splits=scoring,
                    verifier=verifier,
                    capture_item_outcomes=True,
                )
                scored["continuation_replicates"] = collect_scoring_replicates(
                    labeler,
                    scored,
                    directory,
                    [row["continuation"]["seed"] + o for o in offsets],
                )
                scored["selection_h1"] = dict(zip(HEADS, reference_h1(row).tolist()))
                scored["selection_seeds"] = [
                    r["seed"] for r in row.get("continuation_replicates", [])
                ] + [row["continuation"]["seed"]]
                scored["selection_items"] = sorted(selection_items)
                scored["scoring_items"] = sorted(scoring_items)
                write_json(directory / "scored.json", scored)
                measured.append(scored)
        repeated_targets(measured)
        write_json(
            output / "report.json",
            independent_selection_report(
                measured, decisions, forecaster.metadata["utility_weights"]
            ),
        )
    finally:
        backend.restore(saved)
        if backend.state_hash() != saved_hash:
            raise RuntimeError("Independent scoring rollback failed")


if __name__ == "__main__":
    main()
