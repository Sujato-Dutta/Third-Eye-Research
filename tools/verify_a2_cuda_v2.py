"""Independent-node A2 generation, verifier, training and rollback controls."""

import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]


def main():
    from third_eye.cluster import require_gpu_allocation

    require_gpu_allocation()
    import torch
    from third_eye.config import Config
    from third_eye.data.schema import Correction, Example
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.benchmarks import make_verifier
    from third_eye.evaluation.runner import generate_greedy, verify_completions
    from third_eye.io import digest, write_json
    from third_eye.provenance import execution_environment, source_inventory, versions
    from third_eye.training.hf_backend import HFBackend
    from third_eye.updates.corrections import collect_corrections

    from validation_source_identity import model_source_identity

    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    out = ROOT / "runs/a2/validation/v2" / f"cuda_{index}.json"
    if out.exists():
        raise RuntimeError("Validation evidence is immutable")
    models = {}
    verifier = make_verifier()
    for name in ["llama3_2_3b", "qwen3_4b"]:
        cfg = Config.load(ROOT / "runs/a2/configs" / f"{name}.json")
        b = HFBackend(cfg)
        parent, parent_hash = b.snapshot(), b.state_hash()
        baseline, samples = {}, {}
        for task in ["math", "code"]:
            _, splits = load_manifest(
                ROOT / "data/processed/a1" / task / "selection.json"
            )
            for role in ["target_dev", "ood_dev", "retention_dev"]:
                # Retention examples are shared; evaluate them once per model.
                if task == "code" and role == "retention_dev":
                    continue
                examples = splits[role]
                text = generate_greedy(
                    b,
                    examples,
                    cfg.protocol.max_new_tokens,
                    [cfg.protocol.seed] * len(examples),
                )
                verdicts = verify_completions(verifier, examples, text)
                baseline[f"{task}_{role}"] = {"texts": text, "verdicts": verdicts}
            examples = splits["train"][:16]
            prompts = [ex.prompt for ex in examples]
            seeds = [cfg.protocol.seed + 100000 + i * 16 for i in range(len(examples))]
            first = b.generate_sampled_many(prompts, 512, seeds, 0.7)
            second = b.generate_sampled_many(prompts, 512, seeds, 0.7)
            if first != second:
                raise RuntimeError(
                    f"Within-process sampled decoding changed: {name}/{task}"
                )
            samples[task] = {
                "texts": first,
                "verdicts": verify_completions(verifier, examples, first),
            }
        # Diagnostic training fixtures come from archived, verifier-passing
        # training corrections; these produce no scientific labels or scores.
        ref = (
            Path(os.environ["THIRD_EYE_A1_ROOT"])
            / "runs/campaign_a1_20261003/pilots"
            / f"{name}_math_1042"
        )
        row = json.loads((ref / "meta_labels.jsonl").read_text().splitlines()[0])
        pool = json.loads(
            (ref / "states" / row["state_id"] / "correction_pool.json").read_text()
        )
        fixtures = [
            Correction(Example(**item["example"]), item["completion"], item["attempt"])
            for item in pool["items"][:2]
        ]
        if len(fixtures) != 2 or not all(
            verifier.verify(c.example, c.completion) for c in fixtures
        ):
            raise RuntimeError("Invalid diagnostic training fixtures")
        logs = []
        for trial in range(2):
            b.restore(parent)
            log = b.train(fixtures, cfg.protocol.seed + 300000)
            if log["optimizer_steps"] != 50:
                raise RuntimeError("Training verification budget changed")
            logs.append(
                {
                    "adapter_hash": b.state_hash(),
                    "loss_first": log["loss_first"],
                    "loss_last": log["loss_last"],
                }
            )
        if logs[0] != logs[1] or logs[0]["adapter_hash"] == parent_hash:
            raise RuntimeError("Repeated fifty-step training is not deterministic")
        directory = ROOT / "runs/a2/validation/v2" / f"cuda_{index}_{name}_checkpoint"
        b.save_checkpoint(directory, {"purpose": "implementation verification only"})
        b.restore(parent)
        b.load_checkpoint(directory)
        if b.state_hash() != logs[0]["adapter_hash"]:
            raise RuntimeError("Checkpoint reload changed adapter")
        b.restore(parent)
        # Exercise the native eight-attempt collector and terminal statistics.
        _, splits = load_manifest(ROOT / "data/processed/a1/math/selection.json")
        pool8, stats = collect_corrections(
            b, splits["train"][:32], verifier, cfg.protocol, cfg.protocol.seed
        )
        if (
            stats["correction_attempt_limit"] != 8
            or stats["correction_seed_stride"] != 16
            or b.state_hash() != parent_hash
        ):
            raise RuntimeError("A2 collector budget/rollback verification failed")
        models[name] = {
            "config_hash": cfg.fingerprint,
            "parent_hash": parent_hash,
            "baseline": baseline,
            "samples": samples,
            "training": logs[0],
            "pool8": [c.to_dict() for c in pool8],
            "pool8_counts": {
                k: stats[k]
                for k in [
                    "training_prompts",
                    "initial_failures",
                    "verified_corrections",
                    "revision_attempts",
                    "unresolved_failures",
                ]
            },
            "model_source_identity": model_source_identity(cfg, b),
            "attention_implementation": b.model.config._attn_implementation,
        }
        write_json(
            out.with_name(f"cuda_{index}_{name}_partial.json"),
            {
                "status": "model_checks_passed",
                "model": name,
                "result": models[name],
                "execution": execution_environment(),
                "source_tree_sha256": source_inventory(ROOT)["source_tree_sha256"],
            },
        )
        print(json.dumps({"model": name, "verification": "passed"}), flush=True)
        del b, parent
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        out,
        {
            "status": "passed",
            "passed": True,
            "models": models,
            "comparable_sha256": digest(models),
            "execution": execution_environment(),
            "software_versions": versions(),
            "source_tree_sha256": source_inventory(ROOT)["source_tree_sha256"],
        },
    )


if __name__ == "__main__":
    main()
