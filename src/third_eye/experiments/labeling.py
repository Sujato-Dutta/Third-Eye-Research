"""H=2 labels: candidate update followed by a standardized self-update.

Only the candidate's t+1 checkpoint is eligible for commitment. The t+2 branch
is ground truth for the forecaster, never a selectable update or input feature.
"""

from pathlib import Path
import time

from third_eye.evaluation.runner import evaluate
from third_eye.experiments.features import extract_features
from third_eye.io import append_jsonl, digest, publish_jsonl_batch, write_json
from third_eye.updates.candidates import batch_hash, sample_batches
from third_eye.updates.corrections import InsufficientCorrections, collect_corrections


class LabelGenerator:
    def __init__(
        self,
        backend,
        config,
        splits,
        verifier,
        output,
        manifest_hash,
        feature_extractor=extract_features,
    ):
        self.backend, self.config, self.splits = backend, config, splits
        self.verifier, self.output = verifier, Path(output)
        self.manifest_hash, self.feature_extractor = manifest_hash, feature_extractor
        self.output.mkdir(parents=True, exist_ok=True)

    def label_state(self, generation, history=()):
        b, cfg, p = self.backend, self.config, self.config.protocol
        parent, parent_hash = b.snapshot(), b.state_hash()
        seed = p.seed + generation * 1_000_000
        state_id = digest(
            {
                "parent": parent_hash,
                "config": cfg.fingerprint,
                "splits": self.manifest_hash,
                "generation": generation,
            }
        )[:20]
        state_dir = self.output / "states" / state_id
        state_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        records = []
        try:
            baseline = evaluate(b, self.splits, self.verifier, p.max_new_tokens, seed)
            b.save_checkpoint(
                state_dir / "parent", {"state_id": state_id, "generation": generation}
            )
            pool, pool_stats = collect_corrections(
                b, self.splits["train"], self.verifier, p, seed
            )
            write_json(
                state_dir / "correction_pool.json",
                {
                    "state_id": state_id,
                    "stats": pool_stats,
                    "items": [item.to_dict() for item in pool],
                },
            )
            batches = sample_batches(pool, p.candidate_size, p.candidates, seed)
            # Same seed, steps, LR, rank, and optimizer reset for all branches.
            update_seed, continuation_seed = seed + 300_000, seed + 600_000
            for index, batch in enumerate(batches):
                b.restore(parent)
                if b.state_hash() != parent_hash:
                    raise RuntimeError("Candidate did not start from the parent state")
                candidate_id = f"k{index}-{batch_hash(batch)[:10]}"
                directory = state_dir / candidate_id
                directory.mkdir()
                write_json(directory / "batch.json", [item.to_dict() for item in batch])
                features = self.feature_extractor(
                    b,
                    batch,
                    self.splits["retention_dev"],
                    p,
                    update_seed,
                    directory / "probe.jsonl",
                )
                if b.state_hash() != parent_hash:
                    raise RuntimeError("Feature extractor must restore the parent")
                train_log = b.train(
                    batch, update_seed, log_path=directory / "training_t1.jsonl"
                )
                t1 = evaluate(b, self.splits, self.verifier, p.max_new_tokens, seed)
                checkpoint_t1 = directory / "t1"
                t1_hash = b.state_hash()
                b.save_checkpoint(
                    checkpoint_t1,
                    {"state_id": state_id, "candidate_id": candidate_id, "horizon": 1},
                )
                # The procedure/seed is identical; verified examples legitimately
                # vary with the candidate's evolved t+1 model.
                continuation_pool, continuation_stats = collect_corrections(
                    b, self.splits["train"], self.verifier, p, continuation_seed
                )
                continuation = sample_batches(
                    continuation_pool, p.candidate_size, 1, continuation_seed
                )[0]
                write_json(
                    directory / "continuation_batch.json",
                    [item.to_dict() for item in continuation],
                )
                continuation_log = b.train(
                    continuation,
                    continuation_seed,
                    log_path=directory / "training_t2.jsonl",
                )
                t2 = evaluate(b, self.splits, self.verifier, p.max_new_tokens, seed)
                b.save_checkpoint(
                    directory / "t2",
                    {"state_id": state_id, "candidate_id": candidate_id, "horizon": 2},
                )
                delta1, delta2 = t1.delta(baseline), t2.delta(baseline)
                record = {
                    "schema_version": 1,
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "generation": generation,
                    "model": cfg.model.name,
                    "seed": p.seed,
                    "config_hash": cfg.fingerprint,
                    "manifest_hash": self.manifest_hash,
                    "parent_adapter_hash": parent_hash,
                    "candidate_adapter_hash": t1_hash,
                    "batch_hash": batch_hash(batch),
                    "candidate_size": len(batch),
                    "precommit": {
                        "state": baseline.to_dict(),
                        "history": list(history),
                        "features": features,
                        "pool_statistics": pool_stats,
                    },
                    "labels": {
                        "h1": delta1.to_dict(),
                        "h2": delta2.to_dict(),
                        "utility_h1": delta1.utility(p.utility_weights),
                        "utility_h2": delta2.utility(p.utility_weights),
                    },
                    "evaluation": {
                        "parent": baseline.to_dict(),
                        "t1": t1.to_dict(),
                        "t2": t2.to_dict(),
                    },
                    "checkpoints": {
                        "t1": str(checkpoint_t1.resolve()),
                        "t2": str((directory / "t2").resolve()),
                    },
                    "runtime": {
                        "candidate": train_log,
                        "continuation": continuation_log,
                    },
                    "continuation": {
                        "seed": continuation_seed,
                        "batch_hash": batch_hash(continuation),
                        "pool_statistics": continuation_stats,
                    },
                }
                write_json(directory / "record.json", record)
                records.append(record)
            # Publish labels only for a complete matched K=3 state. Partial branch
            # artifacts stay diagnostic; they cannot silently enter meta-training.
            summary = {
                "state_id": state_id,
                "generation": generation,
                "status": "complete",
                "candidates": len(records),
                "seconds": time.perf_counter() - started,
            }
            write_json(state_dir / "summary.json", summary)
            write_json(state_dir / "labels.json", records)
            publish_jsonl_batch(self.output / "meta_labels.jsonl", records)
            append_jsonl(self.output / "ledger.jsonl", summary)
            return records
        except Exception as exc:
            write_json(
                state_dir / "failure.json",
                {
                    "state_id": state_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_candidates": len(records),
                },
            )
            raise
        finally:
            b.restore(parent)
            if b.state_hash() != parent_hash:
                raise RuntimeError("State labeling rollback failed")


def run_trajectory(
    labeler, policy="random", start_generation=0, history=(), selector=None
):
    """An external selector receives only pre-commit fields, never labels."""
    import random

    if policy not in {"random", "greedy_h1", "external"}:
        raise ValueError("Unknown trajectory policy")
    if policy == "external" and selector is None:
        raise ValueError("External policy needs a selector callback")
    cfg, b, output = labeler.config, labeler.backend, labeler.output
    history, accepted = list(history), []
    for generation in range(start_generation, start_generation + cfg.protocol.depth):
        if generation >= 5:
            raise ValueError("T=5 limit exceeded")
        try:
            records = labeler.label_state(generation, history[-3:])
        except InsufficientCorrections as exc:
            append_jsonl(
                output / "ledger.jsonl",
                {
                    "generation": generation,
                    "status": "stopped_insufficient_corrections",
                    "error": str(exc),
                },
            )
            raise
        if policy == "greedy_h1":
            selected = max(
                range(len(records)), key=lambda i: records[i]["labels"]["utility_h1"]
            )
        elif policy == "random":
            selected = random.Random(cfg.protocol.seed + generation).randrange(
                len(records)
            )
        else:
            views = [
                {
                    "candidate_id": r["candidate_id"],
                    "state_id": r["state_id"],
                    "precommit": r["precommit"],
                }
                for r in records
            ]
            selected = selector(views)
            if type(selected) is not int or not 0 <= selected < len(records):
                raise ValueError("Selector must return a candidate index")
        record = records[selected]
        b.load_checkpoint(Path(record["checkpoints"]["t1"]))
        if b.state_hash() != record["candidate_adapter_hash"]:
            raise RuntimeError("Committed adapter differs from selected t+1 checkpoint")
        history.append(
            {
                "generation": generation,
                "accepted_candidate": record["candidate_id"],
                "consequences": record["evaluation"]["t1"],
                "delta": record["labels"]["h1"],
            }
        )
        metadata = {
            "generation": generation + 1,
            "history": history[-3:],
            "policy": policy,
            "state_id": record["state_id"],
            "candidate_id": record["candidate_id"],
            "manifest_hash": labeler.manifest_hash,
        }
        committed = output / "accepted" / f"generation_{generation + 1}"
        b.save_checkpoint(committed, metadata)
        write_json(committed / "resume.json", {"config": cfg.to_dict(), **metadata})
        append_jsonl(
            output / "ledger.jsonl",
            {**metadata, "status": "accepted", "checkpoint": str(committed.resolve())},
        )
        accepted.append(metadata)
    return accepted
