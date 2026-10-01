"""Policy evaluation without the expensive H=2 label-generation tree."""

from pathlib import Path
import random
import time

from third_eye.evaluation.runner import evaluate
from third_eye.experiments.features import extract_features
from third_eye.io import append_jsonl, digest, write_json
from third_eye.updates.candidates import batch_hash, sample_batches
from third_eye.updates.corrections import collect_corrections

POLICIES = (
    "no_update",
    "random",
    "heuristic",
    "greedy_h1",
    "one_step",
    "matched_h1",
    "direct",
    "dynamics",
)


def run_online(
    backend,
    config,
    splits,
    verifier,
    output,
    manifest_hash,
    policy,
    forecaster=None,
    start_generation=0,
    history=(),
    trajectory_id=None,
    prune=False,
    keep_accepted=5,
    generations=None,
    prior_points=(),
):
    """All learned policies see only pre-commit inputs, train one full update.

    Greedy pays for three full H=1 branches. All policies use matched pools and
    reversible feature probes; actual wall time includes their true costs.
    """
    if policy not in POLICIES:
        raise ValueError("Unknown online policy")
    if policy in {"one_step", "matched_h1", "direct", "dynamics"}:
        if forecaster is None or forecaster.metadata["kind"] != policy:
            raise ValueError("Policy and saved forecaster architecture differ")
        if forecaster.metadata["horizon"] != (
            1 if policy in {"one_step", "matched_h1"} else 2
        ):
            raise ValueError("Online policy and forecaster supervision horizon differ")
        if (
            list(config.protocol.utility_weights)
            != forecaster.metadata["utility_weights"]
        ):
            raise ValueError(
                "Forecaster utility weights differ from the frozen protocol"
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    p = config.protocol
    count = p.depth if generations is None else generations
    if not 1 <= count <= p.depth or start_generation + count > 5:
        raise ValueError("Requested trajectory exceeds T=5")
    history = list(history)
    initial = evaluate(backend, splits, verifier, p.max_new_tokens, p.seed)
    points = [
        {
            "generation": start_generation,
            "scores": initial.to_dict(),
            "utility": initial.utility(p.utility_weights),
        }
    ]
    if prior_points:
        if (
            prior_points[-1]["generation"] != start_generation
            or prior_points[-1]["scores"] != initial.to_dict()
        ):
            raise ValueError(
                "Resume trajectory does not match the restored checkpoint's development scores"
            )
        points = list(prior_points)
    for generation in range(start_generation, start_generation + count):
        started = time.perf_counter()
        parent = backend.snapshot()
        parent_hash = backend.state_hash()
        seed = p.seed + generation * 1_000_000
        directory = output / "online" / f"generation_{generation}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            baseline = evaluate(backend, splits, verifier, p.max_new_tokens, seed)
            selected_id = None
            pool_stats = {}
            update_log = {}
            views = []
            if policy != "no_update":
                pool, pool_stats = collect_corrections(
                    backend, splits["train"], verifier, p, seed
                )
                batches = sample_batches(pool, p.candidate_size, p.candidates, seed)
                for k, batch in enumerate(batches):
                    backend.restore(parent)
                    features = extract_features(
                        backend,
                        batch,
                        splits["retention_dev"],
                        p,
                        seed + 300_000,
                        directory / f"probe_{k}.jsonl",
                    )
                    views.append(
                        {
                            "state_id": digest(
                                {
                                    "parent": parent_hash,
                                    "config": config.fingerprint,
                                    "splits": manifest_hash,
                                    "generation": generation,
                                }
                            )[:20],
                            "candidate_id": f"k{k}-{batch_hash(batch)[:10]}",
                            "precommit": {
                                "state": baseline.to_dict(),
                                "history": history[-3:],
                                "features": features,
                                "pool_statistics": pool_stats,
                                "generation": generation,
                            },
                        }
                    )
                    if backend.state_hash() != parent_hash:
                        raise RuntimeError("Online probe failed to restore the parent")
                write_json(directory / "precommit.json", views)
                if forecaster is not None:
                    if views[0]["state_id"] in forecaster.metadata.get(
                        "selection_state_ids", []
                    ):
                        raise ValueError(
                            "Online evaluation state was used to fit or select the forecaster"
                        )
                    if trajectory_id in forecaster.metadata.get(
                        "selection_trajectory_ids", []
                    ):
                        raise ValueError(
                            "Online evaluation trajectory was used to fit or select the forecaster"
                        )
                    selected = forecaster.select(views)
                    write_json(
                        directory / "forecast.json", forecaster.predict(views).tolist()
                    )
                elif policy == "random":
                    selected = random.Random(p.seed + generation).randrange(
                        len(batches)
                    )
                elif policy == "heuristic":
                    selected = min(
                        range(len(batches)),
                        key=lambda i: views[i]["precommit"]["features"][
                            "pre_update_nll"
                        ],
                    )
                else:
                    outcomes = []
                    for k, batch in enumerate(batches):
                        backend.restore(parent)
                        log = backend.train(
                            batch,
                            seed + 300_000,
                            log_path=directory / f"greedy_{k}.jsonl",
                        )
                        outcome = evaluate(
                            backend, splits, verifier, p.max_new_tokens, seed
                        )
                        checkpoint = directory / f"branch_{k}"
                        backend.save_checkpoint(
                            checkpoint,
                            {
                                "generation": generation,
                                "candidate_id": views[k]["candidate_id"],
                            },
                        )
                        outcomes.append(
                            (outcome.utility(p.utility_weights), log, outcome)
                        )
                    selected = max(range(len(outcomes)), key=lambda i: outcomes[i][0])
                    backend.load_checkpoint(directory / f"branch_{selected}")
                    update_log = outcomes[selected][1]
                    write_json(
                        directory / "greedy_cost.json",
                        {"candidate_logs": [o[1] for o in outcomes]},
                    )
                selected_id = views[selected]["candidate_id"]
                write_json(
                    directory / "selected_batch.json",
                    [c.to_dict() for c in batches[selected]],
                )
                if policy != "greedy_h1":
                    backend.restore(parent)
                    update_log = backend.train(
                        batches[selected],
                        seed + 300_000,
                        log_path=directory / "selected_training.jsonl",
                    )
            after = evaluate(backend, splits, verifier, p.max_new_tokens, seed)
            delta = after.delta(baseline)
            history.append(
                {
                    "generation": generation,
                    "accepted_candidate": selected_id,
                    "consequences": after.to_dict(),
                    "delta": delta.to_dict(),
                }
            )
            metadata = {
                "generation": generation + 1,
                "history": history[-3:],
                "policy": policy,
                "mode": "online",
                "forecaster_weights_sha256": forecaster.metadata["weights_sha256"]
                if forecaster
                else None,
                "manifest_hash": manifest_hash,
                "trajectory_id": trajectory_id,
                "candidate_id": selected_id,
            }
            checkpoint = output / "accepted" / f"generation_{generation + 1}"
            backend.save_checkpoint(checkpoint, metadata)
            write_json(
                checkpoint / "resume.json", {"config": config.to_dict(), **metadata}
            )
            elapsed = time.perf_counter() - started
            row = {
                **metadata,
                "status": "accepted",
                "scores": after.to_dict(),
                "delta": delta.to_dict(),
                "utility": after.utility(p.utility_weights),
                "utility_delta": delta.utility(p.utility_weights),
                "seconds": elapsed,
                "pool_statistics": pool_stats,
                "training": update_log,
                "probe_seconds": sum(
                    v["precommit"]["features"].get("probe_seconds", 0) for v in views
                ),
                "checkpoint": str(checkpoint.resolve()),
            }
            append_jsonl(output / "ledger.jsonl", row)
            points.append(
                {
                    "generation": generation + 1,
                    "scores": after.to_dict(),
                    "utility": row["utility"],
                    "seconds": elapsed,
                }
            )
            write_json(
                output / "trajectory.json",
                {
                    "points": points,
                    "policy": policy,
                    "seed": p.seed,
                    "model": config.model.name,
                    "manifest_hash": manifest_hash,
                    "complete": False,
                },
            )
            if prune:
                from third_eye.storage import prune_online_branches

                prune_online_branches(output, generation)
            if keep_accepted < 5:
                from third_eye.storage import prune_accepted

                prune_accepted(output, generation + 1, keep_accepted)
        except Exception as exc:
            backend.restore(parent)
            if backend.state_hash() != parent_hash:
                raise RuntimeError("Online failure rollback failed") from exc
            write_json(
                directory / "failure.json",
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            raise
    result = {
        "points": points,
        "policy": policy,
        "seed": p.seed,
        "model": config.model.name,
        "manifest_hash": manifest_hash,
        "trajectory_id": trajectory_id,
        "complete": True,
    }
    write_json(output / "trajectory.json", result)
    return result
