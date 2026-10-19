"""Additional H2 labels from an unchanged saved M1; never commit an M2."""

from pathlib import Path

from third_eye.io import write_json
from third_eye.updates.candidates import batch_hash, sample_batches
from third_eye.updates.corrections import InsufficientCorrections, collect_corrections


def collect_replicates(labeler, record, directory, seed_offsets):
    """Extend supervision without changing the original H1/H2 observations.

    Offsets are fixed in run.json before collection and shared by all K=3
    candidates. Each draw re-harvests from exactly M1 under the same protocol;
    candidate training and the accepted trajectory are unchanged.
    """
    # Keep evaluation injectable exactly like the original label generator.
    from third_eye.experiments.labeling import evaluate

    backend, config = labeler.backend, labeler.config
    protocol = config.protocol
    original_seed = record["continuation"]["seed"]
    observations = [
        {
            "seed": original_seed,
            "candidate_adapter_hash": record["candidate_adapter_hash"],
            "continuation_adapter_hash": record["continuation_adapter_hash"],
            "evaluation_manifest_hash": record["manifest_hash"],
            "h2": record["labels"]["h2"],
            "continuation_available": record["continuation_available"],
            "optimizer_steps": record["runtime"]["continuation"]["optimizer_steps"],
            "reserved_optimizer_steps": config.training.max_steps,
        }
    ]
    capture = getattr(labeler, "capture_item_outcomes", False)
    if capture:
        observations[0]["items_path"] = record["evaluation_artifacts"]["t2"]
    saved, saved_hash = backend.snapshot(), backend.state_hash()
    try:
        for offset in seed_offsets:
            seed = original_seed + offset
            destination = directory / "continuation_replicates" / str(seed)
            destination.mkdir(parents=True, exist_ok=False)
            backend.load_checkpoint(Path(record["checkpoints"]["t1"]))
            m1_hash = backend.state_hash()
            if m1_hash != record["candidate_adapter_hash"]:
                raise RuntimeError("Repeated continuation did not restore the saved M1")
            pool, stats = collect_corrections(
                backend, labeler.splits["train"], labeler.verifier, protocol, seed
            )
            available = len(pool) >= protocol.candidate_size
            if not available and not protocol.terminal_continuation:
                raise InsufficientCorrections(
                    "Repeated continuation has insufficient verified corrections"
                )
            batch = (
                sample_batches(
                    pool, protocol.candidate_size, 1, seed, protocol.candidate_sampling
                )[0]
                if available
                else ()
            )
            write_json(
                destination / "pool.json",
                {
                    "stats": stats,
                    "items": [c.to_dict() for c in pool],
                    **(
                        {"input_rendering": backend.input_rendering}
                        if hasattr(backend, "input_rendering")
                        else {}
                    ),
                },
            )
            write_json(destination / "batch.json", [c.to_dict() for c in batch])
            if available:
                log = backend.train(
                    batch, seed, log_path=destination / "training.jsonl"
                )
                eval_seed = protocol.seed + record["generation"] * 1_000_000
                outcome = evaluate(
                    backend,
                    labeler.splits,
                    labeler.verifier,
                    protocol.max_new_tokens,
                    eval_seed,
                    **(
                        {"item_output": destination / "evaluation.json"}
                        if capture
                        else {}
                    ),
                ).to_dict()
                h2 = {
                    h: outcome[h] - record["evaluation"]["parent"][h]
                    for h in ("target", "ood", "retention")
                }
            else:
                log = {
                    "optimizer_steps": 0,
                    "seconds": 0.0,
                    "terminal_reason": "correction_scarcity",
                }
                h2 = dict(record["labels"]["h1"])
                if backend.state_hash() != m1_hash:
                    raise RuntimeError("Repeated terminal continuation changed M1")
            if log["optimizer_steps"] != (
                config.training.max_steps if available else 0
            ):
                raise RuntimeError("Repeated continuation optimizer budget differs")
            observation = {
                "seed": seed,
                "candidate_adapter_hash": m1_hash,
                "continuation_adapter_hash": backend.state_hash(),
                "evaluation_manifest_hash": record["manifest_hash"],
                "h2": h2,
                "continuation_available": int(available),
                "optimizer_steps": log["optimizer_steps"],
                "reserved_optimizer_steps": config.training.max_steps,
                "batch_hash": batch_hash(batch) if available else None,
                "pool_statistics": stats,
                "training": log,
            }
            if capture:
                observation["items_path"] = (
                    str((destination / "evaluation.json").resolve())
                    if available
                    else record["evaluation_artifacts"]["t1"]
                )
            backend.save_checkpoint(
                destination / "t2",
                {
                    "state_id": record["state_id"],
                    "candidate_id": record["candidate_id"],
                    "seed": seed,
                    "horizon": 2,
                    "continuation_available": int(available),
                },
            )
            write_json(destination / "observation.json", observation)
            observations.append(observation)
    finally:
        backend.restore(saved)
        if backend.state_hash() != saved_hash:
            raise RuntimeError("Repeated-continuation rollback failed")
    return observations


def collect_scoring_replicates(labeler, record, directory, seeds):
    """Collect independent scoring outcomes; caller supplies disjoint roles.

    This uses the same continuation implementation as label collection. The
    original placeholder observation is discarded, never treated as a draw.
    Caller must measure parent/M1 on the scoring roles before this invocation.
    """
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Scoring needs at least three distinct continuation seeds")
    base = min(seeds) - 1
    if base < 0:
        raise ValueError("Scoring seeds must be positive")
    surrogate = dict(record, continuation={"seed": base})
    observations = collect_replicates(
        labeler, surrogate, directory, [s - base for s in seeds]
    )
    return observations[1:]
