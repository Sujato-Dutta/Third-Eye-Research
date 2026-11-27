"""Prospective matched-batch overlap experiment on fresh Qwen3-4B parents."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from long_horizon_trajectories import saved_config
from next_round_regimes import eligible_pool
from positive_regime_confirmation import configured
from revision_overlap_batches import corrections, matched_batches, profile
from revision_replay import check_shared_inputs
from revision_saved_audit import closed_intervals, paired_data
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.evaluation.final import evaluate_final
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.confirmation import read_scored_items
from third_eye.experiments.regime_revision import unique_batch
from third_eye.io import digest, file_digest, write_json
from third_eye.updates.corrections import collect_corrections

SEEDS = tuple(range(2026112500, 2026112512))
DOSES = ("low", "medium", "high")
ARMS = ("native", "direct")
BASELINES = ("m0", "m1")
ARM_NAMES = tuple(f"{dose}_{arm}" for dose in DOSES for arm in ARMS)
ENDPOINTS = ("low_minus_high_interaction", "first_update", "low_minus_high_native",
             "low_minus_high_direct", "medium_minus_high_interaction",
             "low_minus_medium_interaction", "medium_native_minus_pooled")
MEANINGFUL_EFFECT_PP = 1.0


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_bindings(args):
    return {name: file_digest(path) for name, path in {
        "request": args.request, "original_confirmation": args.confirmation,
        "overlap_code": Path(__file__), "matcher": Path(matched_batches.__code__.co_filename),
        "release_manifest": Path(__file__).with_name("overlap_release.sha256"),
    }.items()}


def planning_precision(work):
    """Historical candidate-to-candidate spread is a planning proxy only."""
    by_seed = {}
    item_hashes = None
    for name in ("dgx_saved_outcomes.npz", "lonestar_saved_outcomes.npz"):
        with np.load(work / name, allow_pickle=False) as bundle:
            meta = json.loads(str(bundle["metadata"]))
            signature = (meta["item_hashes"], meta["input_hashes"])
            if item_hashes not in (None, signature):
                raise ValueError("Planning archives do not share evaluation identities")
            item_hashes = signature
            seeds = meta["seeds"]
            if set(seeds) & by_seed.keys():
                raise ValueError("Duplicate historical parent")
            role_gaps = {}
            for role in meta["item_hashes"]:
                _, _, arms = paired_data(bundle[role])
                role_gaps[role] = arms[:, :, 0].astype(float) - arms[:, :, 1]
            for i, seed in enumerate(seeds):
                by_seed[seed] = {role: roles[i] for role, roles in role_gaps.items()}
    if len(by_seed) != 20:
        raise ValueError("Planning requires all 20 archived parents")
    rows = [by_seed[k] for k in sorted(by_seed)]
    rng = np.random.default_rng(20261125)
    reps = 4000
    results = {}
    for n in (9, len(SEEDS), 15):
        indices = rng.integers(len(rows), size=(reps, n))
        first = rng.integers(3, size=(reps, n))
        second = (first + rng.integers(1, 3, size=(reps, n))) % 3
        means = []
        for role in item_hashes[0]:
            gaps = np.stack([row[role] for row in rows])
            means_for_role = []
            for offset in range(0, reps, 40):
                sl = slice(offset, offset + 40)
                effects = (gaps[indices[sl], first[sl]] - gaps[indices[sl], second[sl]]).mean(axis=1)
                weights = rng.multinomial(effects.shape[1],
                                          np.full(effects.shape[1], 1/effects.shape[1]),
                                          size=len(effects)) / effects.shape[1]
                means_for_role.extend(np.sum(effects * weights, axis=1))
            means.append(means_for_role)
        noise = 100 * np.mean(means, axis=0)
        low, high = np.quantile(noise, [.025, .975])
        results[str(n)] = {"null_interval_pp": [float(low), float(high)],
                           "half_width_pp": float((high-low)/2),
                           "one_pp_detection_proxy": float(np.mean(
                               np.abs(noise + MEANINGFUL_EFFECT_PP) > max(-low, high)))}
    return results


def validate(args, request, splits):
    from transformers import AutoTokenizer
    from third_eye.training.hf_backend import HFBackend

    work = args.output.parent
    smoke = read(work / "probe_smoke.json")
    replay = read(work / "archived_probe_review.json")
    if not smoke["passed"] or not smoke["exact_rollback"] or not replay["passed"]:
        raise ValueError("Expected the successful A100 and archive validation receipts")
    timing = [read(work / "pricing" / f"{seed}.json") for seed in (2026112400, 2026112401)]
    if any(not row["passed"] or not row["timing_only"] or row["accuracy_retained"] for row in timing):
        raise ValueError("Expected both outcomes-blind pricing reports")
    cfg = configured(request, SEEDS[0])
    if cfg.training.micro_batch_size * cfg.training.gradient_accumulation_steps != 8:
        raise ValueError("Pooled control and paired batches require eight presentations per step")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision,
                                              local_files_only=True, use_fast=True)
    if not tokenizer.is_fast or not callable(HFBackend.load_checkpoint):
        raise ValueError("Pinned tokenizer or adapter backend is unavailable")
    coverage = sum(profile(ex.answer) is not None for ex in splits["train"])
    if coverage < 8:
        raise ValueError("No usable arithmetic profiles")
    write_json(args.output / "validation.json", {
        "passed": True, "job_id": os.environ["SLURM_JOB_ID"], "seeds": SEEDS,
        "sources": source_bindings(args), "training_annotated_profiles": coverage,
        "precision_proxy": planning_precision(work), "meaningful_effect_pp": MEANINGFUL_EFFECT_PP,
    })


def check_release(args):
    receipt = read(args.output / "validation.json")
    if not receipt["passed"] or receipt["sources"] != source_bindings(args) or receipt["seeds"] != list(SEEDS):
        raise ValueError("Exact compute-node release validation required")


def make_backend(request, seed):
    import torch
    from third_eye.training.hf_backend import HFBackend

    require_gpu_allocation()
    if torch.cuda.get_device_properties(0).total_memory < 38_000_000_000:
        raise ValueError("A full A100 is required")
    cfg = configured(request, seed)
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    return cfg, backend, verifier


def checkpoint(backend, folder, name, metadata):
    with tempfile.TemporaryDirectory(dir=folder, prefix="checkpoint-") as work:
        staged = Path(work) / name
        backend.save_checkpoint(staged, metadata)
        state = read(staged / "state.json")
        staged.rename(folder / name)
    return state["adapter_hash"]


def train(backend, batch, seed, steps):
    info = backend.train(batch, seed, max_steps=steps)
    if info["optimizer_steps"] != steps or info["examples"] != len(batch):
        raise ValueError("Training budget changed")
    return info


def make_doses(first, future, lengths, seed):
    if not first:
        return {"available": False, "reason": "initial scarcity"}
    matched = matched_batches(first, future, lengths, seed+10_019, shared_doses=True)
    for batch in matched.get("batches", {}).values():
        batch["items"] = [c.to_dict() for c in batch["items"]]
    return matched


def harvest(args, request, splits):
    seed = args.seed
    folder = args.output / "parents" / str(seed)
    done = folder / "harvest.json"
    if done.exists():
        check_harvest(folder, seed)
        return
    folder.mkdir(parents=True, exist_ok=True)
    cfg, backend, verifier = make_backend(request, seed)
    m0 = backend.snapshot()
    m0_hash = backend.state_hash()
    if not (folder / "m0").exists():
        checkpoint(backend, folder, "m0", {"seed": seed})
    elif read(folder / "m0" / "state.json")["adapter_hash"] != m0_hash:
        raise ValueError("Partial run has a different initial adapter")
    first_path = folder / "initial.json"
    if not first_path.exists():
        pool, stats = collect_corrections(backend, splits["train"], verifier, cfg.protocol, seed)
        pool, rejected = eligible_pool(backend, pool)
        first = unique_batch(pool, 8, seed+17) or []
        write_json(first_path, {"batch": [c.to_dict() for c in first], "statistics": stats,
                               "length_rejected": rejected})
    first = corrections(read(first_path)["batch"])
    initial_training = None
    if (folder / "m1").exists():
        backend.load_checkpoint(folder / "m1")
        initial_training = read(folder / "m1" / "state.json")["metadata"]["training"]
    else:
        initial_training = train(backend, first, seed+101, 25) if first else None
        checkpoint(backend, folder, "m1", {"seed": seed, "training": initial_training,
                                           "batch_sha256": digest(read(first_path)["batch"])})
    future_path = folder / "continuation_pool.json"
    if not future_path.exists():
        future, stats = [], {"not_harvested": "initial batch unavailable"}
        if first:
            pool, stats = collect_corrections(backend, splits["train"], verifier, cfg.protocol, seed+10_000)
            future, rejected = eligible_pool(backend, pool)
            stats["length_rejected"] = rejected
        write_json(future_path, {"corrections": [c.to_dict() for c in future], "statistics": stats})
    future = corrections(read(future_path)["corrections"])
    lengths = {c.example.id: sum(label != -100 for label in
                                 backend._encode(c.example.prompt, c.completion)["labels"]) for c in future}
    matched = make_doses(first, future, lengths, seed)
    write_json(done, {"passed": True, "seed": seed, "job_id": os.environ["SLURM_JOB_ID"],
                      "initial_available": bool(first), "matched": matched,
                      "continuation_label_tokens": lengths,
                      "inputs": {path: file_digest(folder / path) for path in
                                 ("initial.json", "continuation_pool.json", "m0/state.json", "m1/state.json")},
                      "training": initial_training, "final_scores_used": False})
    backend.restore(m0)
    print(f"Harvest {seed}: {len(future)} eligible corrections; matched={matched['available']}", flush=True)


def check_harvest(folder, seed):
    record = read(folder / "harvest.json")
    expected = {path: file_digest(folder / path) for path in
                ("initial.json", "continuation_pool.json", "m0/state.json", "m1/state.json")}
    if not record["passed"] or record["seed"] != seed or record["inputs"] != expected or record["final_scores_used"]:
        raise ValueError(f"Pre-scoring harvest changed for {seed}")
    return record


def blind_freeze(args):
    records = {str(seed): check_harvest(args.output / "parents" / str(seed), seed) for seed in SEEDS}
    for seed, record in records.items():
        folder = args.output / "parents" / seed
        initial = read(folder / "initial.json")["batch"]
        future = read(folder / "continuation_pool.json")["corrections"]
        state = read(folder / "m1" / "state.json")
        if (state["metadata"]["batch_sha256"] != digest(initial) or
                make_doses(corrections(initial), corrections(future),
                           record["continuation_label_tokens"], int(seed)) != record["matched"]):
            raise ValueError(f"Batch-matching commitment could not be reproduced for {seed}")
    write_json(args.output / "frozen_harvest.json", {
        "passed": True, "job_id": os.environ["SLURM_JOB_ID"], "sources": source_bindings(args),
        "harvest_hashes": {seed: file_digest(args.output / "parents" / seed / "harvest.json") for seed in records},
        "available": {seed: row["matched"]["available"] for seed, row in records.items()},
        "reasons": {seed: row["matched"].get("reason") for seed, row in records.items()},
    })


def verify_freeze(args, seed, folder):
    frozen_path = args.output / "frozen_harvest.json"
    frozen = read(frozen_path)
    if (not frozen["passed"] or frozen["sources"] != source_bindings(args) or
            frozen["harvest_hashes"][str(seed)] != file_digest(folder / "harvest.json")):
        raise ValueError(f"Frozen batch declaration changed for {seed}")
    return file_digest(frozen_path)


def score_parent(args, request, final):
    seed = args.seed
    folder = args.output / "parents" / str(seed)
    frozen_hash = verify_freeze(args, seed, folder)
    record = check_harvest(folder, seed)
    if (folder / "complete.json").exists():
        if read(folder / "complete.json")["frozen_harvest_sha256"] != frozen_hash:
            raise ValueError("Completed parent belongs to different frozen inputs")
        return
    cfg, backend, verifier = make_backend(request, seed)
    snapshots, states = {}, {}
    for name in BASELINES:
        backend.load_checkpoint(folder / name)
        snapshots[name] = backend.snapshot()
        states[name] = backend.state_hash()
    matched = record["matched"]
    if matched["available"]:
        first = corrections(read(folder / "initial.json")["batch"])
        batches = {dose: corrections(matched["batches"][dose]["items"]) for dose in DOSES}
        randomized = [(f"{dose}_{arm}", "m1" if arm == "native" else "m0", batches[dose], 25)
                      for dose in DOSES for arm in ARMS]
        randomized.append(("pooled", "m0", first+batches["medium"], 50))
        rng = np.random.default_rng(seed+300)
        rng.shuffle(randomized)
        for name, start, batch, steps in randomized:
            if len(batch) != (16 if name == "pooled" else 8) or len({c.example.id for c in batch}) != len(batch):
                raise ValueError("Frozen matched-batch sizes or uniqueness changed")
            backend.restore(snapshots[start])
            if backend.state_hash() != states[start]:
                raise ValueError("Paired update did not restore its assigned starting checkpoint")
            backend.config = replace(cfg, training=replace(cfg.training, max_steps=steps))
            metadata = {"start_hash": states[start], "optimizer_seed": seed+10_031,
                        "corrections_sha256": digest([c.to_dict() for c in batch]),
                        "steps": steps, "examples_seen": steps*8,
                        "label_tokens_seen": (steps*8//len(batch)) * sum(
                            sum(y != -100 for y in backend._encode(c.example.prompt, c.completion)["labels"])
                            for c in batch)}
            path = folder / name / "state.json"
            if path.exists():
                saved = read(path)["metadata"]
                if any(saved[key] != value for key, value in metadata.items()):
                    raise ValueError("Cached update differs from frozen treatment")
                states[name] = read(path)["adapter_hash"]
            else:
                metadata["training"] = train(backend, batch, seed+10_031, steps)
                states[name] = checkpoint(backend, folder, name, metadata)
    commitment = {"frozen_harvest_sha256": frozen_hash,
                  "states": {name: {"adapter_hash": state, "state_sha256": file_digest(folder/name/"state.json")}
                             for name, state in sorted(states.items())}}
    commitment_path = folder / "scoring_commitment.json"
    if commitment_path.exists():
        if read(commitment_path) != commitment:
            raise ValueError("Existing scoring commitment changed")
    else:
        write_json(commitment_path, commitment)
    commit_hash = file_digest(commitment_path)
    scoring = folder / "scoring"
    scoring.mkdir(exist_ok=True)
    scored_by_adapter = {}
    for name in commitment["states"]:
        dest = scoring / name
        if dest.exists():
            read_scored_items(dest, final, states[name], commit_hash)
            scored_by_adapter[states[name]] = dest
            continue
        prior = scored_by_adapter.get(states[name])
        with tempfile.TemporaryDirectory(dir=scoring, prefix="evaluation-") as temp:
            scored = Path(temp) / "outcomes"
            if prior is None:
                backend.config = saved_config(read(folder / name / "state.json")["config"])
                backend.load_checkpoint(folder / name)
                backend.config = cfg
                if backend.state_hash() != states[name]:
                    raise ValueError("Scoring adapter differs from frozen commitment")
                evaluate_final(backend, final, verifier, scored, 512, 0,
                               {"commitment_sha256": commit_hash, "parent": seed, "arm": name})
            else:
                shutil.copytree(prior, scored)
                meta = read(scored / "scores.json")
                write_json(scored / "scores.json", {**meta, "arm": name, "reused_identical_adapter": prior.name})
            scored.rename(dest)
        scored_by_adapter[states[name]] = dest
        print(f"Scored {seed} {name}", flush=True)
    write_json(folder / "complete.json", {"passed": True, "seed": seed,
               "job_id": os.environ["SLURM_JOB_ID"], "frozen_harvest_sha256": frozen_hash,
               "commitment_sha256": commit_hash, "matched_available": matched["available"]})


def review(args, final):
    roles = {role: [] for role in final}
    matched_seeds, dose_means, exposures = [], {}, {}
    for seed in SEEDS:
        folder = args.output / "parents" / str(seed)
        freeze = verify_freeze(args, seed, folder)
        harvest_record = check_harvest(folder, seed)
        done = read(folder / "complete.json")
        path = folder / "scoring_commitment.json"
        if not done["passed"] or done["frozen_harvest_sha256"] != freeze or done["commitment_sha256"] != file_digest(path):
            raise ValueError("Incomplete or changed scored parent")
        commitment = read(path)
        valid = harvest_record["matched"]["available"]
        names = set(BASELINES) | (set(ARM_NAMES) | {"pooled"} if valid else set())
        if set(commitment["states"]) != names:
            raise ValueError("Expected the full assigned scoring inventory")
        if any(file_digest(folder / name / "state.json") != state["state_sha256"]
               for name, state in commitment["states"].items()):
            raise ValueError("Checkpoint metadata differs from the scoring commitment")
        scores = {name: read_scored_items(folder/"scoring"/name, final, state["adapter_hash"], file_digest(path))
                  for name, state in commitment["states"].items()}
        if valid:
            matched_seeds.append(seed)
            dose_means[seed] = {dose: row["mean_similarity"]
                               for dose, row in harvest_record["matched"]["batches"].items()}
            exposures[seed] = {name: read(folder / name / "state.json")["metadata"]["label_tokens_seen"]
                               for name in (*ARM_NAMES, "pooled")}
        for role in roles:
            m0, m1 = (scores[name][role] for name in BASELINES)
            if valid:
                native = {dose: scores[f"{dose}_native"][role] for dose in DOSES}
                direct = {dose: scores[f"{dose}_direct"][role] for dose in DOSES}
                gaps = {dose: (native[dose]-m1) - (direct[dose]-m0) for dose in DOSES}
                rows = [gaps["low"]-gaps["high"], m1-m0, native["low"]-native["high"],
                        direct["low"]-direct["high"], gaps["medium"]-gaps["high"],
                        gaps["low"]-gaps["medium"], native["medium"]-scores["pooled"][role]]
            else:
                zero = np.zeros_like(m0)
                rows = [zero, m1-m0, zero, zero, zero, zero, zero]
            roles[role].append(rows)
    arrays = {role: np.asarray(rows) for role, rows in roles.items()}
    # Scarcity is an identity response for the intended-cohort contrast.
    # The pooled arm has no treatment on scarce parents; report it only when available.
    full = dict(zip(ENDPOINTS[:-1], closed_intervals({r: x[:, :-1] for r, x in arrays.items()})))
    conditional = {}
    if len(matched_seeds) >= 2:
        mask = np.isin(SEEDS, matched_seeds)
        conditional = dict(zip(ENDPOINTS, closed_intervals({r: x[mask] for r, x in arrays.items()})))
    write_json(args.output/"review.json", {
        "passed": True, "job_id": os.environ["SLURM_JOB_ID"], "planned_seeds": SEEDS,
        "matched_seeds": matched_seeds, "dose_mean_similarity": dose_means, "label_token_exposure": exposures,
        "primary_endpoint": ENDPOINTS[0], "intended_cohort": full, "matched_only_descriptive": conditional,
        "by_role": {role: dict(zip(ENDPOINTS[:-1], closed_intervals({role: x[:, :-1]})))
                    for role, x in arrays.items()},
    })


def main():
    require_compute_allocation()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "harvest", "freeze", "score", "review"))
    for name in ("request", "confirmation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    if args.mode != "validate":
        check_release(args)
    if args.mode in ("harvest", "score"):
        if args.seed is None:
            parser.error("GPU worker requires --seed")
        if shutil.disk_usage(args.output).free < 8_000_000_000:
            raise ValueError("Insufficient scratch for retained scoring artifacts")
    request, splits, final = check_shared_inputs(args)
    if args.mode == "validate":
        validate(args, request, splits)
    elif args.mode == "harvest":
        harvest(args, request, splits)
    elif args.mode == "freeze":
        blind_freeze(args)
    elif args.mode == "score":
        score_parent(args, request, final)
    else:
        review(args, final)


if __name__ == "__main__":
    main()
