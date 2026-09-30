"""CLI for pilot and scalable H=2 meta-label trajectories."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.labeling import LabelGenerator, run_trajectory
from third_eye.io import digest, write_json


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
    parser.add_argument("--policy", choices=("random", "greedy_h1"), default="random")
    parser.add_argument(
        "--verifier",
        help="Trusted local module:factory for benchmark/sandbox verifiers",
    )
    parser.add_argument(
        "--resume",
        help="Accepted checkpoint; continue with identical config in a NEW output directory",
    )
    parser.add_argument(
        "--gate2", help="JSON decision artifact with passed=true; required for 8B runs"
    )
    args = parser.parse_args()
    config = Config.load(args.config)
    if config.protocol.status != "frozen" and not args.pilot:
        parser.error(
            "Protocol is a pilot; use --pilot or freeze measured hyperparameters first"
        )
    if "8b" in config.model.name.lower():
        if (
            not args.gate2
            or json.loads(Path(args.gate2).read_text()).get("passed") is not True
        ):
            parser.error("8B scaling is disabled until Gate 2 is documented as passed")
    manifest, splits = load_manifest(args.manifest)
    manifest_hash = digest(manifest)
    verifier = VerifierRegistry.from_spec(args.verifier)
    if (
        any(ex.task == "code" for examples in splits.values() for ex in examples)
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
    if args.resume:
        saved = backend.load_checkpoint(args.resume)
        meta = saved["metadata"]
        if (
            meta.get("manifest_hash") != manifest_hash
            or meta.get("policy") != args.policy
        ):
            parser.error("Resume manifest/policy differs from the accepted checkpoint")
        start_generation, history = meta["generation"], meta["history"]
    if start_generation + config.protocol.depth > 5:
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
            "config": config.to_dict(),
            "config_hash": config.fingerprint,
            "manifest_hash": manifest_hash,
            "resolved_revision": backend.resolved_revision,
            "policy": args.policy,
            "git_commit": git_commit,
            "pilot": args.pilot,
            "resume": args.resume,
            "label_units": "accuracy fractions",
        },
    )
    labeler = LabelGenerator(backend, config, splits, verifier, output, manifest_hash)
    accepted = run_trajectory(labeler, args.policy, start_generation, history)
    write_json(output / "completed.json", {"accepted": accepted, "status": "complete"})
    print(
        f"Completed {len(accepted)} accepted update(s); labels: {output / 'meta_labels.jsonl'}"
    )


if __name__ == "__main__":
    main()
