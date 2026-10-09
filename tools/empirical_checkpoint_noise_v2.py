# ruff: noqa: E402
"""Matched E2 controls load actual retained weights; no historical reconstruction."""

from pathlib import Path
import time
from empirical_checkpoint_v2 import OUT, frozen, passed, arm_deadline, load_noise_input
from empirical_noise_v1 import same_hash, batch, ROLES, HEADS
from third_eye.cluster import require_gpu_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.evaluation.benchmarks import make_verifier
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.io import digest, file_digest, write_json
from third_eye.updates.candidates import batch_hash, sample_batches
from third_eye.updates.corrections import collect_corrections


def collect_items(backend, splits, verifier, protocol, seed):
    before = backend.state_hash()
    result, metrics, variability, texts = {}, {}, {}, {}
    for role, head in zip(ROLES, HEADS):
        examples = splits[role]
        if not examples or len({e.id for e in examples}) != len(examples):
            raise RuntimeError("Unique nonempty evaluation IDs required")
        print(f"E2 item evaluation {role}: {len(examples)}", flush=True)
        completions = generate_greedy(
            backend, examples, protocol.max_new_tokens, [seed] * len(examples)
        )
        verdicts = verify_completions(verifier, examples, completions)
        texts[role] = [
            dict(id=e.id, completion=text) for e, text in zip(examples, completions)
        ]
        result[role] = [
            dict(
                id=e.id,
                prompt_sha256=e.prompt_hash,
                prediction_sha256=digest(text),
                correct=int(v),
            )
            for e, text, v in zip(examples, completions, verdicts)
        ]
        metrics[head] = sum(verdicts) / len(examples)
        code_indices = [i for i, e in enumerate(examples) if e.task == "code"]
        if code_indices:
            exs = [examples[i] for i in code_indices]
            texts = [completions[i] for i in code_indices]
            repeated = [verify_completions(verifier, exs, texts) for _ in range(2)]
            original = [verdicts[i] for i in code_indices]
            variability[role] = dict(
                repeats=3,
                totals=[sum(original), *[sum(v) for v in repeated]],
                fluctuating_items=sum(
                    len({original[i], *[v[i] for v in repeated]}) > 1
                    for i in range(len(exs))
                ),
                items=len(exs),
                identical_prediction_texts=True,
            )
    same_hash(backend, before, "E2 evaluation state preservation")
    return dict(
        adapter_hash=before,
        metrics=metrics,
        items=result,
        fixed_prediction_verifier_repeats=variability,
        prediction_texts=texts,
    )


def difference(measured, original):
    if set(original) != set(HEADS):
        raise RuntimeError("Missing original consequence heads")
    measured["original_recorded_metrics"] = original
    measured["reevaluation_minus_original"] = {
        h: measured["metrics"][h] - original[h] for h in HEADS
    }


def initialize(task, output):
    from third_eye.training.hf_backend import HFBackend

    cfg = Config.load(task["config"])
    _, splits = load_manifest(task["manifest"])
    verifier = make_verifier()
    backend = HFBackend(cfg)
    rows = task["selected_records"]
    if len(rows) != 3 or len({r["batch_hash"] for r in rows}) != 3:
        raise RuntimeError("Complete distinct K=3 required")
    train = {e.id: e for e in splits["train"]}
    for row in rows:
        if (
            row["config_hash"] != cfg.fingerprint
            or row["runtime"]["candidate"]["optimizer_steps"] != 50
        ):
            raise RuntimeError("Native candidate config/budget differs")
        for continuation in (False, True):
            items = batch(task, row, continuation)
            if any(c.example != train.get(c.example.id) for c in items):
                raise RuntimeError(
                    "Saved corrections must be unchanged manifest train examples"
                )
            if items and not all(
                verify_completions(
                    verifier, [c.example for c in items], [c.completion for c in items]
                )
            ):
                raise RuntimeError("Saved correction failed fixed verifier")
    state = Path(task["source"]) / "states" / task["state_id"]
    backend.load_checkpoint(state / "parent")
    if len({r["parent_adapter_hash"] for r in rows}) != 1:
        raise RuntimeError("Candidates lack a shared parent")
    same_hash(backend, rows[0]["parent_adapter_hash"], "actual saved parent")
    return cfg, splits, verifier, backend


def original_m1(backend, task, row, parent, directory):
    backend.load_checkpoint(row["checkpoints"]["t1"])
    same_hash(backend, row["candidate_adapter_hash"], "actual saved M1")
    return backend.snapshot()


def original_m2(backend, task, row, directory):
    backend.load_checkpoint(row["checkpoints"]["t2"])
    same_hash(backend, row["continuation_adapter_hash"], "actual saved M2")
    if not row["continuation_available"]:
        same_hash(backend, row["candidate_adapter_hash"], "actual terminal M2=M1")


def measure(index):
    scope = frozen()
    passed("cpu")
    passed("cuda")
    arm_deadline(scope)
    require_gpu_allocation()
    task = load_noise_input(index)
    output = OUT / "noise" / f"task_{index}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        cfg, splits, verifier, backend = initialize(task, output)
        p = cfg.protocol
        seed = p.seed + task["generation"] * 1_000_000
        parent = backend.snapshot()
        parent_items = collect_items(backend, splits, verifier, p, seed)
        difference(parent_items, task["selected_records"][0]["evaluation"]["parent"])
        write_json(output / "parent_items.json", parent_items)
        results = []
        for row in task["selected_records"]:
            directory = output / row["candidate_id"]
            directory.mkdir()
            m1 = original_m1(backend, task, row, parent, directory)
            one = collect_items(backend, splits, verifier, p, seed)
            difference(one, row["evaluation"]["t1"])
            write_json(directory / "original_m1_items.json", one)
            original_m2(backend, task, row, directory)
            two = collect_items(backend, splits, verifier, p, seed)
            difference(two, row["evaluation"]["t2"])
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
                if items and log["optimizer_steps"] != 50:
                    raise RuntimeError("Fixed-batch optimizer budget differs")
                backend.save_checkpoint(
                    directory / name,
                    dict(
                        condition=name,
                        seed=train_seed,
                        continuation_available=int(bool(items)),
                    ),
                )
                measured = (
                    collect_items(backend, splits, verifier, p, seed)
                    if items
                    else dict(one)
                )
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
            noise_release_sha256=file_digest(OUT / "release.json"),
        )
        write_json(output / "completed.json", record)
    except Exception as exc:
        write_json(
            output / "failure.json", dict(error_type=type(exc).__name__, error=str(exc))
        )
        raise
