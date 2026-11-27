"""Replay short probes on sealed batches and price fresh end-to-end scoring."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from long_horizon_trajectories import copy_checkpoint, saved_config
from next_round_regimes import eligible_pool
from positive_regime_confirmation import configured
from revision_paired_probe import ROLES, checked_inputs, probe, read, score
from revision_saved_audit import closed_intervals
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import evaluate_final, load_final_manifest
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.positive_regime import DECLARATION
from third_eye.experiments.regime_revision import unique_batch
from third_eye.io import digest, file_digest, write_json
from third_eye.updates.corrections import collect_corrections

PROBE_COUNTS = dict(zip(ROLES, (128, 64, 64)))
LAMBDA_GRID = (0.0, .25, .5, 1.0)
TIMING_SEEDS = (2026112400, 2026112401)


def spec(args, seed):
    work = args.request.parent.parent
    ready = args.work
    parent = work / "positive" / "parents" / str(seed) if seed >= 2026110913 else (
        ready / ("migration" if seed == 2026110900 else "replay_migration") / str(seed)
    )
    frozen = read(args.confirmation)
    return SimpleNamespace(
        request=args.request, registry=args.registry, parent=parent,
        parent_sha256=frozen["parent_completed_sha256"][str(seed)],
    )


def select_probe_items(splits, encode):
    out = {}
    for role, count in PROBE_COUNTS.items():
        ordered = sorted(splits[role], key=lambda example: digest([20261124, example.id]))
        usable = []
        for example in ordered:
            try:
                encode(example.prompt, example.answer)
            except ValueError as error:
                if "SFT example exceeds" not in str(error):
                    raise
                continue
            usable.append(example)
            if len(usable) == count:
                break
        if len(usable) != count:
            raise ValueError(f"Frozen probe requires {count} complete {role} items")
        out[role] = usable
    return out


def check_shared_inputs(args):
    request = read(args.request)
    for field in ("config", "manifest", "final_manifest"):
        if file_digest(request[field]) != request[field+"_sha256"]:
            raise ValueError(f"Frozen input changed: {field}")
    if read(args.confirmation)["declaration"] != DECLARATION:
        raise ValueError("Expected the original confirmation declaration")
    _, splits = load_manifest(request["manifest"])
    _, final = load_final_manifest(request["final_manifest"], request["manifest"])
    if (len(splits["train"]) != DECLARATION["training_prompts"] or
            {role: len(items) for role, items in final.items()} != DECLARATION["evaluation_items"]):
        raise ValueError("Harvest/evaluation workload differs")
    return request, splits, final


def validate(args):
    from transformers import AutoTokenizer
    from third_eye.training.tokenization import encode_completion
    from third_eye.training.hf_backend import HFBackend

    request, splits, _ = check_shared_inputs(args)
    cfg = configured(request, TIMING_SEEDS[0])
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, local_files_only=True, use_fast=True,
    )
    dev = select_probe_items(splits, lambda prompt, answer: encode_completion(
        tokenizer, prompt, answer, cfg.training.max_sequence_length, cfg.model.chat_kwargs,
    ))
    for seed in args.seeds:
        archive = spec(args, seed)
        checked_inputs(archive)
        for name in ("m0", "m1"):
            source = archive.parent / "checkpoints" / name
            with tempfile.TemporaryDirectory(prefix="third-eye-replay-validate-") as temp:
                state = copy_checkpoint(source, Path(temp) / "adapter")
                if saved_config(state["config"]).fingerprint != state["config_hash"]:
                    raise ValueError("Archived config fingerprint changed")
    assert callable(HFBackend.load_checkpoint)
    return {
        "passed": True, "job_id": os.environ["SLURM_JOB_ID"], "validated_seeds": args.seeds,
        "source_sha256": file_digest(__file__), "probe_counts": PROBE_COUNTS,
        "probe_items_sha256": digest({role: [x.id for x in items] for role, items in dev.items()}),
        "request_sha256": file_digest(args.request), "registry_sha256": file_digest(args.registry),
        "confirmation_sha256": file_digest(args.confirmation),
    }


def checked_validation(args):
    validated = read(args.validated)
    expected = {"source_sha256": file_digest(__file__), "request_sha256": file_digest(args.request),
                "registry_sha256": file_digest(args.registry),
                "confirmation_sha256": file_digest(args.confirmation), "probe_counts": PROBE_COUNTS}
    if not validated["passed"] or any(validated[key] != value for key, value in expected.items()):
        raise ValueError("Matching compute validation is required")
    return validated


def replay(args):
    from third_eye.training.hf_backend import HFBackend

    started = time.perf_counter()
    validated = checked_validation(args)
    if args.seed not in validated["validated_seeds"]:
        raise ValueError("This archive seed has not passed input validation")
    archive = spec(args, args.seed)
    splits, batches = checked_inputs(archive)
    original = read(archive.parent / "checkpoints" / "m0" / "state.json")
    cfg = saved_config(original["config"])
    backend = HFBackend(cfg)
    snapshots, hashes = {}, {}
    for name in ("m0", "m1"):
        with tempfile.TemporaryDirectory(prefix="third-eye-replay-adapter-") as temp:
            path = Path(temp) / "adapter"
            restored = copy_checkpoint(archive.parent / "checkpoints" / name, path)
            backend.config = saved_config(restored["config"])
            backend.load_checkpoint(path)
        snapshots[name], hashes[name] = backend.snapshot(), backend.state_hash()
    backend.config = cfg
    dev = select_probe_items(splits, backend._encode)
    probe_ids = digest({r: [x.id for x in items] for r, items in dev.items()})
    if probe_ids != validated["probe_items_sha256"]:
        raise ValueError("GPU probe item selection differs from validation")
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")

    def losses(items):
        return np.asarray([backend.measure_loss([example]) for example in items])

    passed = {}
    for name in ("m0", "m1"):
        backend.restore(snapshots[name])
        items = dev["target_dev"]
        completions = generate_greedy(backend, items, 512, [0]*len(items))
        passed[name] = np.asarray(verify_completions(verifier, items, completions), dtype=bool)
    fixed = [item for item, earned in zip(dev["target_dev"], ~passed["m0"] & passed["m1"]) if earned]
    backend.restore(snapshots["m0"])
    base_old = losses(fixed)
    backend.restore(snapshots["m1"])
    base_current = {role: losses(items) for role, items in dev.items()}
    candidates = []
    seal = read(archive.parent / "commitment.json")
    for draw, batch in enumerate(batches):
        seed = seal["states"][f"d{draw}_native"]["optimizer_seed"]
        after, new = probe(backend, snapshots["m1"], hashes["m1"], batch, seed,
                           lambda: {r: losses(items) for r, items in dev.items()})
        gains = {r: float(np.mean(base_current[r]-after[r])) for r in ROLES}
        gain = float(np.mean(list(gains.values())))
        penalty, old = 0.0, None
        if fixed:
            refit, old = probe(backend, snapshots["m0"], hashes["m0"], batch, seed, lambda: losses(fixed))
            _, penalty = score(gain, base_old-refit, np.ones(len(fixed), dtype=bool), 0.0)
        candidates.append({"draw": draw, "gain": gain, "penalty": penalty, "role_proxy_delta": gains,
                           "probe_on_m1": new, "probe_on_m0": old})
    return {"passed": True, "seed": args.seed, "job_id": os.environ["SLURM_JOB_ID"],
            "fixed_target_items": len(fixed), "probe_counts": PROBE_COUNTS, "probe_items_sha256": probe_ids,
            "candidates": candidates, "archived_parent_sha256": archive.parent_sha256,
            "outcomes_loaded": False, "elapsed_seconds": time.perf_counter()-started}


def price(args):
    from third_eye.training.hf_backend import HFBackend

    checked_validation(args)
    request, splits, final = check_shared_inputs(args)
    started = time.perf_counter()
    stamp = started
    cfg = configured(request, args.seed)
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    timeline = {"model_load_seconds": time.perf_counter()-stamp}
    for phase, delta in (("initial", 0), ("continuation", 10_000)):
        stamp = time.perf_counter()
        pool, _ = collect_corrections(backend, splits["train"], verifier, cfg.protocol, args.seed+delta)
        pool, rejected = eligible_pool(backend, pool)
        batch = unique_batch(pool, 8, args.seed+delta+(17 if not delta else 19))
        timeline[phase] = {"harvest_seconds": time.perf_counter()-stamp,
                           "eligible_corrections": len(pool), "length_rejected": rejected,
                           "update_available": batch is not None}
        if batch is not None:
            training = backend.train(batch, args.seed+delta+(101 if not delta else 31), max_steps=25)
            if training["optimizer_steps"] != 25:
                raise ValueError("Unexpected timing update budget")
            timeline[phase]["train_seconds"] = training["seconds"]
        write_json(args.output.with_name(args.output.stem+"_progress.json"), timeline)
    stamp = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="third-eye-price-") as folder:
        evaluate_final(backend, final, verifier, Path(folder)/"scoring", 512, 0,
                       {"timing_only": True, "timing_seed": args.seed})
    timeline["one_full_checkpoint_seconds"] = time.perf_counter()-stamp
    return {"passed": True, "job_id": os.environ["SLURM_JOB_ID"], "seed": args.seed,
            "timing_only": True, "accuracy_retained": False, "evaluation_counts": DECLARATION["evaluation_items"],
            "timeline": timeline, "elapsed_seconds": time.perf_counter()-started}


def archive_review(args):
    roles, parents = tuple(DECLARATION["evaluation_items"]), DECLARATION["seeds"]
    values = {}
    signature = None
    for path in args.outcomes:
        with np.load(path, allow_pickle=False) as bundle:
            meta = json.loads(str(bundle["metadata"]))
            current = (meta["item_hashes"], meta["input_hashes"])
            if signature not in (None, current):
                raise ValueError("Archived outcomes have different item bindings")
            signature = current
            for i, seed in enumerate(meta["seeds"]):
                if seed in values:
                    raise ValueError("Duplicate archived outcome")
                values[seed] = {r: bundle[r][i, 2::2].astype(float) for r in roles}
    if sorted(values) != parents:
        raise ValueError("Expected the entire archived cohort")
    reports = [read(args.work/"replay"/f"{p}.json") for p in parents]
    if (not all(r["passed"] and r["seed"] == p and not r["outcomes_loaded"]
                for p, r in zip(parents, reports)) or
            len({r["probe_items_sha256"] for r in reports}) != 1):
        raise ValueError("All archived probes must use identical development items and sealed parents")
    scores = np.asarray([[row["gain"] for row in r["candidates"]] for r in reports])
    penalties = np.asarray([[row["penalty"] for row in r["candidates"]] for r in reports])
    utility = np.asarray([[values[p][r].mean(axis=1) for r in roles] for p in parents]).mean(axis=1)
    grid_choices = (scores[:, None, :] - np.asarray(LAMBDA_GRID)[None, :, None]*penalties[:, None, :]).argmax(axis=2)
    predicted = utility[np.arange(len(parents))[:, None], grid_choices]
    # Pick lambda on other seeds' sealed outcomes. The held-out seed uses only probes to select its batch.
    selected_grid = (predicted.sum(axis=0)-predicted).argmax(axis=1)
    baseline = grid_choices[:, 0]
    chose = grid_choices[np.arange(len(parents)), selected_grid]
    effects = {r: np.asarray([[values[p][r][a]-values[p][r][b]]
                              for p, a, b in zip(parents, chose, baseline)]) for r in roles}
    result = closed_intervals(effects)[0]
    retention = closed_intervals({"retention_test": effects["retention_test"]})[0]
    return {"passed": True, "parents": len(parents), "development_only": True,
            "same_candidates_same_panel_ablation": True, "lambda_grid": LAMBDA_GRID,
            "lambda_by_seed": dict(zip(parents, (LAMBDA_GRID[i] for i in selected_grid))),
            "changed_choices": int(np.count_nonzero(chose != baseline)),
            "utility_vs_single_checkpoint": result, "retention_vs_single_checkpoint": retention,
            "fixed_target_items": {r["seed"]: r["fixed_target_items"] for r in reports}}


def main():
    require_compute_allocation()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "replay", "price", "review"))
    for name in ("work", "request", "registry", "confirmation", "output", "validated"):
        parser.add_argument(f"--{name}", type=Path, required=(name != "validated"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--outcomes", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode in ("replay", "price"):
        require_gpu_allocation()
    if args.mode == "price" and args.seed not in TIMING_SEEDS:
        parser.error("Unknown fresh timing seed")
    outputs = {"validate": validate, "replay": replay, "price": price, "review": archive_review}
    result = outputs[args.mode](args)
    write_json(args.output, result)
    print(f"{args.mode} complete: {args.output.name}", flush=True)


if __name__ == "__main__":
    main()
