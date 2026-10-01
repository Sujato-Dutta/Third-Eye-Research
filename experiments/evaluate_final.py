"""Score the base or a sealed adapter on untouched benchmark test sets."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.cluster import require_gpu_allocation
from third_eye.config import Config
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.evaluation.final import load_final_manifest, evaluate_final
from third_eye.io import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--checkpoint",
        help="Omit to evaluate the frozen base model with its initial adapter",
    )
    p.add_argument("--selection-manifest", required=True)
    p.add_argument("--final-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gate2")
    p.add_argument(
        "--verifier",
        default=os.environ.get("THIRD_EYE_VERIFIER"),
        help="Trusted module:factory for the cluster-approved verifier",
    )
    a = p.parse_args()
    cfg = Config.load(a.config)
    require_gpu_allocation(cfg.model.device)
    if cfg.protocol.status != "frozen":
        p.error("Final evaluation requires a frozen protocol")
    if "8b" in cfg.model.name.lower() and (
        not a.gate2 or json.loads(Path(a.gate2).read_text()).get("passed") is not True
    ):
        p.error("8B evaluation requires a passed Gate 2 artifact")
    manifest, splits = load_final_manifest(a.final_manifest, a.selection_manifest)
    from third_eye.training.hf_backend import HFBackend

    backend = HFBackend(cfg)
    metadata = {}
    if a.checkpoint:
        metadata = backend.load_checkpoint(a.checkpoint)["metadata"]
    result = evaluate_final(
        backend,
        splits,
        VerifierRegistry.from_spec(
            a.verifier or "third_eye.evaluation.benchmarks:make_verifier"
        ),
        a.output,
        cfg.protocol.max_new_tokens,
        cfg.protocol.seed,
        {
            "config_hash": cfg.fingerprint,
            "final_manifest_hash": digest(manifest),
            "checkpoint": a.checkpoint,
            "model": cfg.model.name,
            "policy": metadata.get("policy", "no_update"),
            "resolved_revision": backend.resolved_revision,
            "utility_weights": list(cfg.protocol.utility_weights),
        },
    )
    print(json.dumps(result["scores"]))


if __name__ == "__main__":
    main()
