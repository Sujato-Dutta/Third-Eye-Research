"""Bounded development-only recipe pilots, before prospective confirmation."""
import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.benchmarks import audit_overlap, canonical_prompt, MATH_SUBJECTS
from third_eye.data.schema import Correction, Example, ROLES
from third_eye.data.splits import create_manifest, load_manifest
from third_eye.evaluation.final import load_final_manifest
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.regime_revision import RECIPES, unique_batch
from third_eye.io import digest, file_digest, publish_jsonl_batch, write_json
from third_eye.provenance import source_inventory
from third_eye.updates.corrections import collect_corrections

DECLARATION = {
    "version": "next_round_20261106",
    "additional_su_cap": 200, "first_stage_su_cap": 55,
    "recipes": list(RECIPES), "models": ["llama-3.2-3b-instruct", "qwen3-4b"],
    "development_seeds": {"llama-3.2-3b-instruct": list(range(2026110600, 2026110604)),
                          "qwen3-4b": list(range(2026110700, 2026110704))},
    "development_items": {"target_dev": 256, "ood_dev": 256, "retention_dev": 512},
    "evaluation_items": {"target_test": 1319, "ood_test": 500, "retention_test": 1024},
    "positive_screen_g1": .01, "retention_screen_minimum": -.005,
    "screening_uses_final_outcomes": False,
    "cost_pilot_recipe": "r0", "cost_pilot_future_draws": 3,
    "cost_pilot_future_scoring_items_per_role": 16,
    "automatic_confirmation": False, "forecaster_fits": 0,
    "cost_pilot_seconds": 21600,
    "corrections": "eight attempts; verifier passing; complete SFT target fits1024; unique problem IDs",
}


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, validation=False):
    require_compute_allocation()
    request = read(path)
    if (request["declaration"] != DECLARATION or
            request["source_tree_sha256"] != source_inventory()["source_tree_sha256"]):
        raise ValueError("Frozen source or declaration differs")
    if validation:
        result = read(request["validation"])
        if not result["passed"] or result["source_tree_sha256"] != request["source_tree_sha256"]:
            raise ValueError("Matching compute-node validation required")
    if file_digest(request["model_sources"]) != request["model_sources_sha256"]:
        raise ValueError("Frozen model-source mapping changed")
    for task in request["reference_tasks"]:
        for key in ("config", "manifest", "final_manifest"):
            if file_digest(task[key]) != task[key+"_sha256"]:
                raise ValueError("Frozen reference input changed: "+key)
    return request


def ordered(examples, salt):
    return sorted(examples, key=lambda e: (digest([20261106, salt, e.id, e.prompt_hash]), e.id))


def fill(existing, pool, count, excluded, salt):
    result = list(existing)
    if len(result) == count:
        return result
    if len(result) > count:
        raise ValueError("Existing items exceed fixed count")
    used = set(excluded) | {canonical_prompt(x) for x in result}
    for example in ordered(pool, salt):
        if example.id in {x.id for x in result} or canonical_prompt(example) in used:
            continue
        result.append(example)
        used.add(canonical_prompt(example))
        if len(result) == count:
            break
    if len(result) != count:
        raise ValueError("Insufficient independent items for frozen manifest")
    return result


def source_rows(repo, config, split, revision):
    from datasets import load_dataset
    return load_dataset(repo, config, split=split, revision=revision)


def gsm_examples(revision, split, role):
    repo = "openai/gsm8k"
    return [Example(
        f"gsm8k:{split}:{i}",
        row["question"]+"\nSolve step by step. End with #### followed by the numeric answer.",
        row["answer"], role, "math", metadata={"benchmark": "gsm8k",
            "source_repository": repo, "source_revision": revision,
            "source_split": split, "source_prompt": row["question"]})
        for i, row in enumerate(source_rows(repo, "main", split, revision))]


def math_examples(revision, exclusions=None):
    from third_eye.data.benchmarks import boxed_answer
    repo = "EleutherAI/hendrycks_math"
    output = []
    for subject in MATH_SUBJECTS:
        for i, row in enumerate(source_rows(repo, subject, "train", revision)):
            identifier = f"math:train:{row.get('type', subject)}:{i}"
            try:
                answer = boxed_answer(row["solution"])
                if not answer.strip():
                    raise ValueError("MATH reference has an empty boxed answer")
                if not row["problem"].strip():
                    raise ValueError("MATH source has an empty problem")
            except ValueError as error:
                if exclusions is not None:
                    exclusions.append({"id": identifier, "reason": str(error)})
                continue
            output.append(Example(
                identifier,
                row["problem"]+"\nSolve step by step and put the final answer in \\boxed{}.",
                answer, "ood_dev", "math",
                str(row.get("level", "default")), {
                    "benchmark": "math", "verifier": "symbolic_math",
                    "subject": row.get("type", subject), "source_repository": repo,
                    "source_revision": revision, "source_split": "train",
                    "source_prompt": row["problem"]}))
    return output


def mmlu_examples(revision, split, role):
    repo = "cais/mmlu"
    output = []
    for i, row in enumerate(source_rows(repo, "all", split, revision)):
        prompt = row["question"]+"\n"+"\n".join(
            f"{chr(65+j)}. {choice}" for j, choice in enumerate(row["choices"]))
        output.append(Example(f"mmlu:{split}:{row.get('subject','all')}:{i}",
            prompt+"\nReturn only the answer letter A, B, C, or D.",
            chr(65+row["answer"]), role, "exact", metadata={
                "benchmark": "mmlu", "verifier": "multiple_choice",
                "source_repository": repo, "source_revision": revision,
                "source_split": split, "source_prompt": prompt}))
    return output


def prepare(request):
    """Outcome-blind expanded manifests; model loading is not needed."""
    output = Path(request["output"]) / "inputs"
    output.mkdir(parents=True, exist_ok=False)
    task = request["reference_tasks"][0]
    _, dev = load_manifest(task["manifest"])
    _, old_final = load_final_manifest(task["final_manifest"], task["manifest"])
    revisions = {e.metadata["source_repository"]: e.metadata["source_revision"]
                 for rows in [*dev.values(), *old_final.values()] for e in rows}
    train = dev["train"]
    # Block the entire final pools before filling development, not only the
    # historical scored subset (MATH train contains some MATH-500 questions).
    full_target = gsm_examples(revisions["openai/gsm8k"], "test", "target_test")
    full_ood = []
    repo = "HuggingFaceH4/MATH-500"
    for i, row in enumerate(source_rows(repo, None, "test", revisions[repo])):
        full_ood.append(Example(
            f"math500:test:{i}",
            row["problem"]+"\nSolve step by step and put the final answer in \\boxed{}.",
            row["answer"], "ood_test", "math", str(row.get("level", "default")), {
                "benchmark": "math500", "verifier": "symbolic_math",
                "source_repository": repo, "source_revision": revisions[repo],
                "source_split": "test", "source_prompt": row["problem"]}))
    blocked = {canonical_prompt(e) for e in train}
    blocked.update(canonical_prompt(e) for xs in old_final.values() for e in xs)
    blocked.update(canonical_prompt(e) for e in [*full_target, *full_ood])
    dev["target_dev"] = fill(dev["target_dev"],
        gsm_examples(revisions["openai/gsm8k"], "train", "target_dev"),
        256, blocked, "target")
    invalid_references = []
    dev["ood_dev"] = fill(dev["ood_dev"],
        math_examples(revisions["EleutherAI/hendrycks_math"], invalid_references),
        256, blocked, "ood")
    dev["retention_dev"] = fill(dev["retention_dev"],
        mmlu_examples(revisions["cais/mmlu"], "validation", "retention_dev"),
        512, blocked, "retention")
    final = {
        "target_test": full_target,
        "ood_test": full_ood,
        "retention_test": fill(old_final["retention_test"],
            mmlu_examples(revisions["cais/mmlu"], "test", "retention_test"),
            1024, {canonical_prompt(e) for xs in dev.values() for e in xs}, "retention_final"),
    }
    audit_overlap({**dev, **final})
    for role, count in DECLARATION["evaluation_items"].items():
        if len(final[role]) != count:
            raise ValueError("Official evaluation pool size differs from declaration")
    for role, examples in {**dev, **final}.items():
        publish_jsonl_batch(output/(role+".jsonl"), [asdict(e) for e in examples])
    selection = create_manifest({r: output/(r+".jsonl") for r in ROLES}, output/"selection.json")
    write_json(output/"final.json", {"schema_version": 1,
        "selection_hash": digest(selection), "splits": {
            r: {"path": r+".jsonl", "sha256": file_digest(output/(r+".jsonl")),
                "count": len(xs)} for r, xs in final.items()}})
    model_checks = verify_models(request)
    write_json(output/"completed.json", {"passed": True, "revisions": revisions,
        "excluded_invalid_math_references": invalid_references,
        "model_checks": model_checks,
        "selection_sha256": file_digest(output/"selection.json"),
        "final_sha256": file_digest(output/"final.json"),
        "prior_scored_item_counts": {r: len(xs) for r, xs in old_final.items()},
        "outcomes_generated": False, "training_ids_unchanged": True})


def verify_models(request):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from safetensors import safe_open
    from third_eye.training.sources import resolve_source
    from third_eye.training.tokenization import frozen_chat_kwargs

    checks = []
    for task in request["reference_tasks"]:
        cfg = Config.load(task["config"])
        name, revision, proof = resolve_source(cfg.model)
        folder = Path(name) if Path(name).is_dir() else Path(snapshot_download(
            name, revision=revision, local_files_only=True))
        AutoConfig.from_pretrained(str(folder), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(folder), use_fast=True,
                                                  local_files_only=True)
        if not tokenizer.is_fast:
            raise ValueError("Fast tokenizer required before GPU allocation")
        frozen_chat_kwargs(tokenizer, cfg.model.chat_kwargs)
        index = folder/"model.safetensors.index.json"
        names = set(read(index)["weight_map"].values()) if index.exists() else {"model.safetensors"}
        hashes = {}
        for filename in sorted(names):
            weight = folder/filename
            hashes[filename] = file_digest(weight)
            if proof and filename in proof["verified_weight_sha256"]:
                expected = proof["verified_weight_sha256"][filename]
                expected = expected["sha256"] if isinstance(expected, dict) else expected
                if hashes[filename] != expected:
                    raise ValueError("Verified model weights changed")
            with safe_open(str(weight), framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    raise ValueError("Empty model weight shard")
        checks.append({"model": cfg.model.name, "revision": cfg.model.revision,
                       "weights_sha256": hashes, "fast_tokenizer": tokenizer.is_fast})
    return checks


def development_scores(backend, splits, verifier, output):
    output.mkdir(parents=True, exist_ok=False)
    scores = []
    started = time.perf_counter()
    from third_eye.io import append_jsonl
    for role in ("target_dev", "ood_dev", "retention_dev"):
        examples = splits[role]
        texts = generate_greedy(backend, examples, 512, [0]*len(examples))
        verdicts = verify_completions(verifier, examples, texts)
        for ex, text, passed in zip(examples, texts, verdicts):
            append_jsonl(output/"items.jsonl", {
                "id": ex.id, "role": role, "completion": text, "passed": bool(passed)})
        scores.append(sum(verdicts)/len(examples))
    write_json(output/"scores.json", {"scores": scores,
        "seconds": time.perf_counter()-started, "adapter_hash": backend.state_hash()})
    return scores


def eligible_pool(backend, pool):
    accepted, rejected = [], []
    for correction in pool:
        try:
            backend._encode(correction.example.prompt, correction.completion)
        except ValueError as error:
            if "SFT example exceeds" not in str(error):
                raise
            rejected.append(correction.example.id)
        else:
            accepted.append(correction)
    return accepted, rejected


def pilot(request, index, parent_index=0):
    require_gpu_allocation()
    import numpy as np
    import torch
    from third_eye.training.hf_backend import HFBackend
    model = DECLARATION["models"][index]
    task = next(x for x in request["reference_tasks"] if x["stream"] == model+"/math")
    cfg = Config.load(task["config"])
    seed = DECLARATION["development_seeds"][model][parent_index]
    cfg = replace(cfg, training=replace(cfg.training, max_steps=25),
        protocol=replace(cfg.protocol, seed=seed, depth=1))
    root = Path(request["output"])
    inputs = root/"inputs"
    prepared = read(inputs/"completed.json")
    if (not prepared["passed"] or file_digest(inputs/"selection.json") !=
            prepared["selection_sha256"] or file_digest(inputs/"final.json") !=
            prepared["final_sha256"]):
        raise ValueError("Frozen expanded development inputs changed")
    _, splits = load_manifest(inputs/"selection.json")
    folder = root/"development"/model/str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    parent = backend.snapshot()
    original = backend.state_hash()
    fixtures = [Correction(e, e.answer, 1) for e in splits["train"][:8]]
    if not all(verify_completions(verifier, [c.example for c in fixtures],
                                 [c.completion for c in fixtures])):
        raise ValueError("Training-only compatibility fixtures fail verification")
    compatibility = backend.train(fixtures, seed+1, max_steps=1)
    backend.restore(parent)
    if backend.state_hash() != original:
        raise ValueError("Compatibility check failed exact parent rollback")
    write_json(folder/"compatibility.json", {
        "passed": True, "fixture_training": compatibility, "exact_rollback": True,
        "research_update": False})
    backend.save_checkpoint(folder/"parent", {"purpose": "development_cost_pilot"})
    parent_scores = development_scores(backend, splits, verifier, folder/"scoring/parent")
    pool, stats = collect_corrections(backend, splits["train"], verifier, cfg.protocol, seed)
    pool, rejected = eligible_pool(backend, pool)
    write_json(folder/"pool.json", {"stats": stats, "sft_length_rejections": rejected,
        "corrections": [c.to_dict() for c in pool]})
    records, r0 = [], None
    for recipe in RECIPES:
        batch = unique_batch(pool, recipe["batch_size"], seed+17)
        record = {"model": model, "seed": seed, "recipe": recipe["id"],
                  "available": batch is not None}
        if batch is not None:
            backend.config = replace(cfg, training=replace(cfg.training,
                learning_rate=recipe["learning_rate"], max_steps=recipe["steps"]))
            backend.restore(parent)
            train = backend.train(batch, seed+101, max_steps=recipe["steps"])
            branch = folder/"recipes"/recipe["id"]
            backend.save_checkpoint(branch/"m1", {"recipe": recipe, "seed": seed})
            scores = development_scores(backend, splits, verifier, branch/"scoring")
            record.update({"training": train, "development_scores": scores,
                "development_head_gains": (np.asarray(scores)-parent_scores).tolist(),
                "batch_ids": [c.example.id for c in batch]})
            if recipe["id"] == "r0":
                r0 = backend.snapshot()
        records.append(record)
        write_json(folder/"development_records.json", {"records": records})
    development_elapsed = time.perf_counter()-started
    write_json(folder/"development_timing.json", {
        "elapsed_seconds": development_elapsed, "model": model, "seed": seed,
        "source_tree_sha256": request["source_tree_sha256"],
        "development_complete": len(records) == 4,
        "final_outcomes_generated": False})
    timing = []
    if r0 is not None and parent_index == 0:
        recipe = RECIPES[0]
        backend.config = replace(cfg, training=replace(cfg.training,
            learning_rate=recipe["learning_rate"], max_steps=recipe["steps"]))
        tiny = {role: ordered(splits[role], "cost_"+role)[:16]
                for role in ("target_dev", "ood_dev", "retention_dev")}
        for draw in range(3):
            remaining = DECLARATION["cost_pilot_seconds"] - (time.perf_counter()-started)
            expected = max(600., stats["total_seconds"]*1.75)
            if remaining < expected:
                break
            native_seed = seed+10_000+draw*100_000
            backend.restore(r0)
            future, future_stats = collect_corrections(
                backend, splits["train"], verifier, cfg.protocol, native_seed)
            future, rejected = eligible_pool(backend, future)
            batch = unique_batch(future, 8, native_seed+19)
            draw_record = {"draw": draw, "harvest": future_stats,
                           "available": batch is not None,
                           "sft_length_rejections": rejected}
            for arm, snapshot in (("native", r0), ("direct", parent)):
                backend.restore(snapshot)
                training = backend.train(batch, native_seed+31, max_steps=10) if batch else None
                dest = folder/"cost_continuations"/str(draw)/arm
                backend.save_checkpoint(dest/"adapter", {"arm": arm, "draw": draw})
                scores = development_scores(backend, tiny, verifier, dest/"scoring")
                draw_record[arm] = {"training": training, "scores": scores}
            write_json(folder/"cost_continuations"/str(draw)/"donor.json",
                       {"seed": native_seed, "batch": [c.to_dict() for c in batch] if batch else []})
            timing.append(draw_record)
            write_json(folder/"continuation_timings.json", {"draws": timing})
    write_json(folder/"completed.json", {
        "passed": True, "development_complete": len(records)==4,
        "future_timing_complete": len(timing)==3, "model": model, "seed": seed,
        "elapsed_seconds": time.perf_counter()-started, "records": records,
        "development_elapsed_seconds": development_elapsed,
        "continuation_draws": timing, "final_outcomes_generated": False,
        "decision": "REVIEW_MEASURED_COST_BEFORE_REMAINING_DEVELOPMENT"})
    del backend
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "pilot"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--parent-index", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    request = checked(args.request, validation=True)
    os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
    if args.command == "prepare":
        prepare(request)
    else:
        if args.index is None:
            parser.error("pilot requires --index")
        pilot(request, args.index, args.parent_index)


if __name__ == "__main__":
    main()
