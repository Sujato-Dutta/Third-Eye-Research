"""CLI for pilot and scalable H=2 meta-label trajectories."""

import argparse
import json
import os
import random
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.config import Config
from third_eye.cluster import require_gpu_allocation, require_compute_allocation
from third_eye.data.splits import load_manifest
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.labeling import LabelGenerator, run_trajectory
from third_eye.experiments.online import POLICIES, run_online
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory, versions, execution_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Allow an explicitly unfrozen pilot protocol",
    )
    parser.add_argument("--policy", choices=POLICIES, default="random")
    parser.add_argument("--mode", choices=("labels", "online"), default="labels")
    parser.add_argument(
        "--capture-item-outcomes",
        action="store_true",
        help="Label mode: retain original per-item verifier outcomes for paired uncertainty analysis",
    )
    parser.add_argument(
        "--continuation-seed-offsets",
        type=int,
        nargs="*",
        default=[],
        help="Label mode: extra predeclared seeds relative to the original continuation seed; use at least two",
    )
    parser.add_argument(
        "--generations",
        type=int,
        choices=range(1, 6),
        help="Updates in this invocation, bounded by frozen depth and remaining T<=5",
    )
    parser.add_argument(
        "--pool-order-seed",
        type=int,
        help="Stress test: deterministically shuffle only the training prompt order",
    )
    parser.add_argument(
        "--forecaster", help="Saved forecaster directory for a learned online policy"
    )
    parser.add_argument(
        "--prune-branches",
        action="store_true",
        help="After commitment retain accepted adapters, labels and logs; prune branch checkpoints",
    )
    parser.add_argument(
        "--keep-accepted",
        type=int,
        choices=range(1, 6),
        default=5,
        help="Keep this many most recent accepted checkpoints",
    )
    parser.add_argument(
        "--verifier",
        default=os.environ.get("THIRD_EYE_VERIFIER"),
        help="Trusted local module:factory for benchmark/sandbox verifiers",
    )
    parser.add_argument(
        "--resume",
        help="Accepted checkpoint; continue with identical config in a NEW output directory",
    )
    parser.add_argument(
        "--gate2", help="JSON decision artifact with passed=true; required for 8B runs"
    )
    parser.add_argument(
        "--scale-confirmation",
        help="Source-bound empirical scale-study authorization; not a forecasting result",
    )
    args = parser.parse_args()
    if args.mode == "labels" and args.policy not in {"random", "greedy_h1"}:
        parser.error(
            "Label collection uses random or greedy_h1; learned policies use --mode online"
        )
    if (
        args.policy in {"one_step", "matched_h1", "direct", "dynamics", "expected"}
        and not args.forecaster
    ):
        parser.error("Learned policies require --forecaster")
    offsets = args.continuation_seed_offsets
    if args.capture_item_outcomes and args.mode != "labels":
        parser.error("Item outcome capture currently requires label mode")
    if offsets and (
        args.mode != "labels"
        or len(offsets) < 2
        or len(set(offsets)) != len(offsets)
        or min(offsets) <= 0
    ):
        parser.error(
            "Repeated labels need >=2 distinct positive continuation seed offsets in label mode"
        )
    config = Config.load(args.config)
    require_compute_allocation()
    require_gpu_allocation(config.model.device)
    if config.protocol.status != "frozen" and not args.pilot:
        parser.error(
            "Protocol is a pilot; use --pilot or freeze measured hyperparameters first"
        )
    if "8b" in config.model.name.lower():
        if args.scale_confirmation:
            from third_eye.experiments.scale import validate_authorization

            validate_authorization(
                args.scale_confirmation, args.config, args.manifest,
                mode=args.mode, policy=args.policy, generations=args.generations,
                offsets=offsets, capture=args.capture_item_outcomes,
            )
        elif (
            not args.gate2
            or json.loads(Path(args.gate2).read_text()).get("passed") is not True
        ):
            parser.error("8B scaling is disabled until Gate 2 is documented as passed")
    manifest, splits = load_manifest(args.manifest)
    manifest_hash = digest(manifest)
    if args.pool_order_seed is not None:
        splits["train"] = list(splits["train"])
        random.Random(args.pool_order_seed).shuffle(splits["train"])
        manifest_hash = digest(
            {"manifest": manifest, "pool_order_seed": args.pool_order_seed}
        )
    verifier = VerifierRegistry.from_spec(
        args.verifier or "third_eye.evaluation.benchmarks:make_verifier"
    )
    if (
        any(ex.task == "code" for examples in splits.values() for ex in examples)
        and not os.environ.get("THIRD_EYE_SANDBOX_CONFIG")
        and not args.verifier
    ):
        parser.error(
            "Code splits require --verifier with an isolated code execution implementation"
        )
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new/empty; resume into a new directory")
    from third_eye.training.hf_backend import HFBackend

    backend = HFBackend(config)
    start_generation, history = 0, []
    trajectory_id = digest(str(output.resolve()))[:20]
    prior_points = []
    forecaster = None
    if args.mode == "online" and args.forecaster:
        from third_eye.forecasting.training import Forecaster

        forecaster = Forecaster.load(args.forecaster)
    if args.resume:
        saved = backend.load_checkpoint(args.resume)
        meta = saved["metadata"]
        if (
            meta.get("manifest_hash") != manifest_hash
            or meta.get("policy") != args.policy
            or meta.get("mode", args.mode) != args.mode
        ):
            parser.error("Resume manifest/policy differs from the accepted checkpoint")
        start_generation, history = meta["generation"], meta["history"]
        trajectory_id = meta.get("trajectory_id", trajectory_id)
        if args.mode == "labels":
            # Older checkpoints may store collection settings only in run.json.
            previous_run = Path(args.resume).resolve().parent.parent / "run.json"
            prior_settings = (
                json.loads(previous_run.read_text(encoding="utf-8"))
                if previous_run.exists()
                else {}
            )
            prior_offsets = meta.get(
                "continuation_seed_offsets",
                prior_settings.get("continuation_seed_offsets", []),
            )
            prior_capture = meta.get(
                "capture_item_outcomes",
                prior_settings.get("capture_item_outcomes", False),
            )
            if prior_offsets != offsets or prior_capture != args.capture_item_outcomes:
                parser.error(
                    "Resume continuation grid/item capture differs from the accepted checkpoint"
                )
        if args.mode == "online":
            if args.policy == "expected" and meta.get(
                "forecaster_metadata_hash"
            ) != digest(forecaster.metadata):
                parser.error(
                    "Resume expected-value model/decision metadata differs from the accepted checkpoint"
                )
            if meta.get("forecaster_weights_sha256") != (
                forecaster.metadata["weights_sha256"] if forecaster else None
            ):
                parser.error(
                    "Resume forecaster weights differ from the accepted checkpoint"
                )
            previous = Path(args.resume).resolve().parent.parent / "trajectory.json"
            if not previous.exists():
                parser.error(
                    "Online resume requires the original trajectory history artifact"
                )
            prior = json.loads(previous.read_text(encoding="utf-8"))
            prior_points = [
                p for p in prior["points"] if p["generation"] <= start_generation
            ]
    generations = args.generations or min(config.protocol.depth, 5 - start_generation)
    if (
        generations < 1
        or generations > config.protocol.depth
        or start_generation + generations > 5
    ):
        parser.error("Requested trajectory exceeds T=5")
    output.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = "uncommitted"
    write_json(
        output / "run.json",
        {
            "schema_version": 1,
            "trajectory_id": trajectory_id,
            "mode": args.mode,
            "pool_order_seed": args.pool_order_seed,
            "forecaster": args.forecaster,
            "verifier": args.verifier
            or "third_eye.evaluation.benchmarks:make_verifier",
            "config": config.to_dict(),
            "config_hash": config.fingerprint,
            "manifest_hash": manifest_hash,
            "resolved_revision": backend.resolved_revision,
            "model_source": backend.source_proof,
            "policy": args.policy,
            "git_commit": git_commit,
            "pilot": args.pilot,
            "resume": args.resume,
            "invocation_generations": generations,
            "label_units": "accuracy fractions",
            "continuation_seed_offsets": offsets,
            "capture_item_outcomes": args.capture_item_outcomes,
            **({"scale_confirmation_authorization_sha256": file_digest(args.scale_confirmation)}
               if args.scale_confirmation else {}),
            "software_versions": versions(),
            "execution": execution_environment(),
            **source_inventory(),
        },
    )
    if args.mode == "online":
        result = run_online(
            backend,
            config,
            splits,
            verifier,
            output,
            manifest_hash,
            args.policy,
            forecaster,
            start_generation,
            history,
            trajectory_id,
            args.prune_branches,
            args.keep_accepted,
            generations,
            prior_points,
        )
        write_json(
            output / "completed.json",
            {
                "status": "complete",
                "mode": "online",
                "generations": len(result["points"]) - 1,
            },
        )
        print(
            f"Completed online {args.policy} trajectory: {output / 'trajectory.json'}"
        )
        return
    labeler = LabelGenerator(
        backend,
        config,
        splits,
        verifier,
        output,
        manifest_hash,
        trajectory_id=trajectory_id,
        continuation_seed_offsets=offsets,
        capture_item_outcomes=args.capture_item_outcomes,
    )
    accepted = run_trajectory(
        labeler,
        args.policy,
        start_generation,
        history,
        prune=args.prune_branches,
        keep_accepted=args.keep_accepted,
        generations=generations,
    )
    write_json(
        output / "completed.json",
        {
            "accepted": accepted,
            "status": "correction_scarcity"
            if len(accepted) < generations
            else "complete",
            "completed_states": len(accepted),
            "requested_generations": generations,
        },
    )
    print(
        f"Completed {len(accepted)} accepted update(s); labels: {output / 'meta_labels.jsonl'}"
    )


if __name__ == "__main__":
    main()
