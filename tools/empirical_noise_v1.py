# ruff: noqa: E402
"""Isolated exact A2 replay and matched noise measurements on allocated GPUs."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_common_v1 import BASE, checked, prepared
from third_eye.cluster import require_gpu_allocation
from third_eye.config import Config
from third_eye.data.schema import Correction, Example
from third_eye.data.splits import load_manifest
from third_eye.evaluation.benchmarks import make_verifier
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.io import digest, file_digest, write_json
from third_eye.updates.candidates import batch_hash, sample_batches
from third_eye.updates.corrections import collect_corrections

ROLES = ("target_dev", "ood_dev", "retention_dev")
HEADS = ("target", "ood", "retention")


def frozen():
    scope = checked()
    release = json.loads((BASE / "noise_release.json").read_text())
    if file_digest(BASE / "release.json") != release["cached_release_sha256"]:
        raise RuntimeError("Cached release changed")
    for name, sha in release["files"].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError(f"Noise release changed: {name}")
    return scope


def same_hash(backend, expected, stage):
    actual = backend.state_hash()
    if actual != expected:
        raise RuntimeError(f"Exact replay failed at {stage}: {actual} != {expected}")
    return actual


def batch(task, row, continuation=False):
    name = "continuation_batch.json" if continuation else "batch.json"
    path = (
        Path(task["source"]) / "states" / row["state_id"] / row["candidate_id"] / name
    )
    values = json.loads(path.read_text())
    items = tuple(
        Correction(Example(**v["example"]), v["completion"], v["attempt"])
        for v in values
    )
    expected = row["continuation"]["batch_hash"] if continuation else row["batch_hash"]
    if items and (
        len(items) != 2
        or len({c.example.id for c in items}) != 2
        or batch_hash(items) != expected
    ):
        raise RuntimeError("Saved batch no longer matches original verified examples")
    if not items and (not continuation or row["continuation_available"]):
        raise RuntimeError("An available update has no examples")
    return items


def collect_items(backend, splits, verifier, protocol, seed):
    before = backend.state_hash()
    result, metrics = {}, {}
    for role, head in zip(ROLES, HEADS):
        examples = splits[role]
        if not examples or len({ex.id for ex in examples}) != len(examples):
            raise RuntimeError("Item evaluation needs unique, nonempty role IDs")
        print(f"Item evaluation {role}: {len(examples)}", flush=True)
        completions = generate_greedy(
            backend, examples, protocol.max_new_tokens, [seed] * len(examples)
        )
        verdicts = verify_completions(verifier, examples, completions)
        result[role] = [
            dict(
                id=ex.id,
                prompt_sha256=ex.prompt_hash,
                prediction_sha256=digest(text),
                correct=int(passed),
            )
            for ex, text, passed in zip(examples, completions, verdicts)
        ]
        metrics[head] = sum(verdicts) / len(examples)
    same_hash(backend, before, "evaluation must preserve adapter")
    return dict(adapter_hash=before, metrics=metrics, items=result)


def exact_metrics(measured, expected, stage):
    if set(expected) != set(HEADS) or any(
        measured["metrics"][h] != expected[h] for h in HEADS
    ):
        raise RuntimeError(
            f"Original aggregate evaluation did not reproduce at {stage}"
        )


def initialize(task, output):
    from third_eye.training.hf_backend import HFBackend

    cfg = Config.load(task["config"])
    _, splits = load_manifest(task["manifest"])
    verifier = make_verifier()
    backend = HFBackend(cfg)
    rows = task["selected_records"]
    if len(rows) != 3 or len({r["batch_hash"] for r in rows}) != 3:
        raise RuntimeError("Need the original distinct K=3 candidate compositions")
    training_ids = {ex.id: ex for ex in splits["train"]}
    # Bind saved corrections to the original train partition and recheck the verifier.
    for row in task["ancestor_records"] + rows:
        for continuation in (False, True):
            items = batch(task, row, continuation)
            if any(c.example != training_ids.get(c.example.id) for c in items):
                raise RuntimeError(
                    "Replay correction differs from manifest training example"
                )
            if not all(
                verify_completions(
                    verifier, [c.example for c in items], [c.completion for c in items]
                )
            ):
                raise RuntimeError(
                    "Saved correction no longer passes the fixed verifier"
                )
    for row in task["ancestor_records"]:
        same_hash(backend, row["parent_adapter_hash"], "ancestor parent")
        backend.train(
            batch(task, row),
            row["runtime"]["candidate"]["seed"],
            log_path=output / f"ancestor_{row['generation']}.jsonl",
        )
        same_hash(backend, row["candidate_adapter_hash"], "accepted ancestor")
    same_hash(backend, rows[0]["parent_adapter_hash"], "selected parent")
    if any(r["parent_adapter_hash"] != backend.state_hash() for r in rows):
        raise RuntimeError("Candidates have different parents")
    backend.save_checkpoint(output / "parent", {"supplement": "E1", "replayed": True})
    return cfg, splits, verifier, backend


def original_m1(backend, task, row, parent, output):
    backend.restore(parent)
    same_hash(backend, row["parent_adapter_hash"], "candidate start")
    log = backend.train(
        batch(task, row),
        row["runtime"]["candidate"]["seed"],
        log_path=output / "original_t1_training.jsonl",
    )
    if log["optimizer_steps"] != 50:
        raise RuntimeError("Replay optimizer budget differs")
    same_hash(backend, row["candidate_adapter_hash"], "original M1")
    backend.save_checkpoint(output / "original_m1", {"replayed": True})
    return backend.snapshot()


def original_m2(backend, task, row, output):
    items = batch(task, row, True)
    if row["continuation_available"]:
        log = backend.train(
            items,
            row["continuation"]["seed"],
            log_path=output / "original_t2_training.jsonl",
        )
        if log["optimizer_steps"] != 50:
            raise RuntimeError("Continuation optimizer budget differs")
    same_hash(backend, row["continuation_adapter_hash"], "original M2")
    backend.save_checkpoint(output / "original_m2", {"replayed": True})


def verification():
    frozen()
    require_gpu_allocation()
    tasks = prepared()["noise_tasks"]
    output = BASE / "gpu_verification"
    output.mkdir(exist_ok=False)
    results = []
    try:
        for task in [t for t in tasks if t["generation"] == 0]:
            directory = output / str(task["index"])
            directory.mkdir()
            cfg, splits, verifier, backend = initialize(task, directory)
            parent = backend.snapshot()
            row = task["selected_records"][0]
            seed = cfg.protocol.seed + task["generation"] * 1_000_000
            baseline = collect_items(backend, splits, verifier, cfg.protocol, seed)
            exact_metrics(baseline, row["evaluation"]["parent"], "verification parent")
            write_json(directory / "parent_items.json", baseline)
            original_m1(backend, task, row, parent, directory)
            one = collect_items(backend, splits, verifier, cfg.protocol, seed)
            exact_metrics(one, row["evaluation"]["t1"], "verification M1")
            write_json(directory / "m1_items.json", one)
            original_m2(backend, task, row, directory)
            two = collect_items(backend, splits, verifier, cfg.protocol, seed)
            exact_metrics(two, row["evaluation"]["t2"], "verification M2")
            write_json(directory / "m2_items.json", two)
            from third_eye.evaluation.runner import evaluate

            subset = {r: splits[r][:4] for r in ROLES}
            measured = collect_items(backend, subset, verifier, cfg.protocol, seed)
            repeated = collect_items(backend, subset, verifier, cfg.protocol, seed)
            if measured != repeated:
                raise RuntimeError("Identical greedy evaluations did not reproduce")
            comparison = evaluate(
                backend, subset, verifier, cfg.protocol.max_new_tokens, seed
            )
            exact_metrics(measured, comparison.to_dict(), "aligned collector")
            results.append(
                dict(
                    stream=task["stream"],
                    task_index=task["index"],
                    parent_hash=row["parent_adapter_hash"],
                    m1_hash=row["candidate_adapter_hash"],
                    m2_hash=backend.state_hash(),
                    collector_matches=True,
                )
            )
            del backend, parent
            gc.collect()
            import torch

            torch.cuda.empty_cache()
        if len(results) != 4:
            raise RuntimeError("All four stream replay checks required")
        write_json(
            output / "passed.json",
            dict(
                status="passed",
                streams=results,
                noise_release_sha256=file_digest(BASE / "noise_release.json"),
            ),
        )
    except Exception as exc:
        write_json(
            output / "failure.json", dict(error_type=type(exc).__name__, error=str(exc))
        )
        raise


def measure(index):
    scope = frozen()
    require_gpu_allocation()
    passed = json.loads((BASE / "gpu_verification/passed.json").read_text())
    if passed["noise_release_sha256"] != file_digest(BASE / "noise_release.json"):
        raise RuntimeError("GPU validation release differs")
    task = prepared()["noise_tasks"][index]
    output = BASE / "noise" / f"task_{index}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        cfg, splits, verifier, backend = initialize(task, output)
        p = cfg.protocol
        seed = p.seed + task["generation"] * 1_000_000
        parent = backend.snapshot()
        parent_items = collect_items(backend, splits, verifier, p, seed)
        exact_metrics(
            parent_items, task["selected_records"][0]["evaluation"]["parent"], "parent"
        )
        write_json(output / "parent_items.json", parent_items)
        results = []
        for row in task["selected_records"]:
            directory = output / row["candidate_id"]
            directory.mkdir()
            m1 = original_m1(backend, task, row, parent, directory)
            one = collect_items(backend, splits, verifier, p, seed)
            exact_metrics(one, row["evaluation"]["t1"], "M1")
            write_json(directory / "original_m1_items.json", one)
            original_m2(backend, task, row, directory)
            two = collect_items(backend, splits, verifier, p, seed)
            exact_metrics(two, row["evaluation"]["t2"], "M2")
            write_json(directory / "original_m2_items.json", two)
            conditions = {"original_m1": one, "original_m2": two}
            for name, start, items, train_seed in [
                (
                    "optimizer_m1",
                    parent,
                    batch(task, row),
                    row["runtime"]["candidate"]["seed"]
                    + scope["fixed_optimizer_seed_offset"],
                ),
                (
                    "optimizer_m2",
                    m1,
                    batch(task, row, True),
                    row["continuation"]["seed"] + scope["fixed_optimizer_seed_offset"],
                ),
            ]:
                backend.restore(start)
                log = (
                    backend.train(
                        items, train_seed, log_path=directory / f"{name}_training.jsonl"
                    )
                    if items
                    else dict(optimizer_steps=0, reserved_optimizer_steps=50)
                )
                backend.save_checkpoint(
                    directory / name,
                    dict(
                        condition=name,
                        seed=train_seed,
                        continuation_available=int(bool(items)),
                    ),
                )
                measured = collect_items(backend, splits, verifier, p, seed)
                measured.update(
                    training=log,
                    seed=train_seed,
                    continuation_available=int(bool(items)),
                )
                if not items:
                    same_hash(
                        backend,
                        row["candidate_adapter_hash"],
                        "terminal optimizer repeat",
                    )
                write_json(directory / f"{name}_items.json", measured)
                conditions[name] = measured
            for offset in scope["fresh_continuation_seed_offsets"]:
                name = f"fresh_m2_{offset}"
                fresh_seed = row["continuation"]["seed"] + offset
                backend.restore(m1)
                same_hash(
                    backend, row["candidate_adapter_hash"], "fresh continuation parent"
                )
                pool, stats = collect_corrections(
                    backend, splits["train"], verifier, p, fresh_seed
                )
                write_json(
                    directory / f"{name}_pool.json",
                    dict(stats=stats, items=[c.to_dict() for c in pool]),
                )
                if (
                    not stats["harvest_complete"]
                    or stats["correction_attempt_limit"] != 8
                    or stats["correction_seed_stride"] != 16
                ):
                    raise RuntimeError("Fresh continuation harvesting budget differs")
                available = len(pool) >= p.candidate_size
                items = (
                    sample_batches(
                        pool, p.candidate_size, 1, fresh_seed, p.candidate_sampling
                    )[0]
                    if available
                    else ()
                )
                write_json(
                    directory / f"{name}_batch.json", [c.to_dict() for c in items]
                )
                if available:
                    log = backend.train(
                        items, fresh_seed, log_path=directory / f"{name}_training.jsonl"
                    )
                    if log["optimizer_steps"] != 50:
                        raise RuntimeError("Fresh continuation training budget differs")
                else:
                    same_hash(
                        backend,
                        row["candidate_adapter_hash"],
                        "terminal fresh continuation",
                    )
                    log = dict(
                        optimizer_steps=0,
                        reserved_optimizer_steps=50,
                        terminal_reason="correction_scarcity",
                    )
                backend.save_checkpoint(
                    directory / name,
                    dict(
                        condition=name,
                        seed=fresh_seed,
                        continuation_available=int(available),
                    ),
                )
                measured = (
                    collect_items(backend, splits, verifier, p, seed)
                    if available
                    else dict(one)
                )
                measured.update(
                    training=log,
                    seed=fresh_seed,
                    continuation_available=int(available),
                    batch_hash=batch_hash(items) if available else None,
                )
                write_json(directory / f"{name}_items.json", measured)
                conditions[name] = measured
            results.append(
                dict(
                    candidate_id=row["candidate_id"],
                    batch_hash=row["batch_hash"],
                    conditions=conditions,
                )
            )
        record = dict(
            status="complete",
            index=index,
            stream=task["stream"],
            generation=task["generation"],
            trajectory_id=task["trajectory_id"],
            state_id=task["state_id"],
            parent=parent_items,
            candidates=results,
            seconds=time.perf_counter() - started,
            noise_release_sha256=file_digest(BASE / "noise_release.json"),
        )
        write_json(output / "completed.json", record)
    except Exception as exc:
        write_json(
            output / "failure.json", dict(error_type=type(exc).__name__, error=str(exc))
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify", "measure"])
    parser.add_argument("--index", type=int, choices=range(12))
    args = parser.parse_args()
    if args.mode == "measure" and args.index is None:
        parser.error("Measurement needs a fixed task index")
    verification() if args.mode == "verify" else measure(args.index)


if __name__ == "__main__":
    main()
