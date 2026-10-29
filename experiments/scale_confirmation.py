"""Serial, frozen eight-parent/model scale sensitivity on DGX allocations."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finish_confirmation import run_task
from forecasting_pilots import inspect_task
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.schema import Correction
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import load_final_manifest
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.confirmation import checkpoints, read_scored_items
from third_eye.experiments.scale import (
    DECLARATION, MODELS, actions, compact_checkpoints, summarize,
)
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.updates.corrections import collect_corrections


def read(path):
    return json.loads(Path(path).read_text())


def checked(path):
    require_compute_allocation()
    request = read(path)
    sha = source_inventory()["source_tree_sha256"]
    validation = read(request["validation"])
    if request["source_tree_sha256"] != sha or not validation["passed"] or validation["source_tree_sha256"] != sha:
        raise ValueError("Matching complete CPU source validation required")
    for name in ("registry", "catalog"):
        if file_digest(request[name]) != request[name + "_sha256"]:
            raise ValueError("Frozen scale input differs: " + name)
    return request


def usage():
    return int(subprocess.run(["du", "-x", "-B1", "-s", str(Path.home())],
                             check=True, capture_output=True, text=True).stdout.split()[0])


def headroom(extra=2_500_000_000):
    used = usage()
    if used + extra > 50_000_000_000:
        raise ValueError("50GB home policy lacks required model/output headroom")
    return used


def evict_weights(folder, weights, root):
    folder = Path(folder).resolve()
    folder.relative_to((root / "models").resolve())
    evicted = []
    for name, expected in weights.items():
        path = folder / name
        path.resolve().relative_to(folder)
        if not name.endswith(".safetensors"):
            continue
        if not path.exists():
            continue
        if path.is_symlink() or file_digest(path) != expected:
            raise ValueError("Refuse cache eviction with an unexpected weight hash")
        path.unlink()
        evicted.append(name)
    return evicted


def rotate(request):
    root, output = Path(request["root"]), Path(request["output"])
    output.mkdir(exist_ok=True)
    controls = Path(request["controls_output"])
    done = read(controls / "paired_controls_analysis/completed.json")
    if not done["passed"] or done["parents"] != 16 or file_digest(controls / "paired_controls_analysis/report.json") != done["report_sha256"]:
        raise ValueError("Complete verified controls required before rotating their caches")
    before = usage()
    registry = read(request["registry"])
    archived = []
    for task in registry["tasks"]:
        parent = controls / "parents" / task["stream"] / str(task["seed"])
        archived.append({"stream": task["stream"], "seed": task["seed"],
                         "files": compact_checkpoints(parent)})
    old = read(root / "prepared.json")
    removed = {}
    for model, proof in old["verified_models"].items():
        removed[model] = evict_weights(proof["runtime_directory"],
                                      {n: x["sha256"] for n, x in proof["verified_files"].items()
                                       if n.endswith(".safetensors")}, root)
    write_json(output / "rotation_completed.json", {
        "passed": True, "before_bytes": before, "after_bytes": usage(),
        "archived_adapters": archived, "evicted_recoverable_base_weights": removed,
        "controls_report_sha256": done["report_sha256"],
        "all_adapter_bytes_preserved": True, "scientific_outcomes_unchanged": True,
    })


def prepare(request, model):
    from huggingface_hub import hf_hub_download

    root, output = Path(request["root"]), Path(request["output"]) / model
    if not read(Path(request["output"]) / "rotation_completed.json")["passed"]:
        raise ValueError("Allocated storage rotation required")
    output.mkdir(exist_ok=False)
    identity, catalog = MODELS[model], read(request["catalog"])
    source = catalog[identity["repository"]]
    official = catalog[identity["official"]]
    if source["revision"] != identity["source_revision"] or official["revision"] != identity["revision"]:
        raise ValueError("Pinned model revision differs")
    weights = {n: v for n, v in official["files"].items() if n.endswith(".safetensors")}
    if not weights or any(source["files"].get(n, {}).get("sha256") != v["sha256"] for n, v in weights.items()):
        raise ValueError("Public source weights must match every official shard")
    if model == "llama3.1-8b":
        previous = Path(request["output"]) / "qwen3-8b"
        done = read(previous / "evidence/completed.json")
        if not done["passed"] or file_digest(previous / "evidence/report.json") != done["report_sha256"]:
            raise ValueError("Complete first 8B model required before cache swap")
        proof = read(previous / "model_source.json")
        removed = evict_weights(proof["runtime_directory"], proof["verified_weight_sha256"], root)
        write_json(output / "cache_swap.json", {"previous_model": "qwen3-8b", "removed_weights": removed})
    names = set(weights) | {n for n in source["files"] if n in {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "model.safetensors.index.json", "vocab.json", "merges.txt",
    } or "chat_template" in n}
    headroom(sum(source["files"][n]["bytes"] for n in names) + 2_500_000_000)
    folder = root / "models/scale" / model
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for name in sorted(names):
        item = source["files"][name]
        headroom(item["bytes"] + 500_000_000)
        path = Path(hf_hub_download(identity["repository"], name,
                                   revision=identity["source_revision"], local_dir=folder, token=False))
        if item["sha256"]:
            verified = file_digest(path) == item["sha256"]
        else:
            blob = path.read_bytes()
            verified = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest() == item["git_blob"]
        if not verified or path.stat().st_size != item["bytes"]:
            raise ValueError("Pinned model file differs: " + name)
        if name not in weights:
            metadata[name] = {"sha256": file_digest(path), "repository": identity["repository"], "revision": identity["source_revision"]}
    proof = {"official": identity["official"], "original_revision": identity["revision"],
             "source_repository": identity["repository"], "source_revision": identity["source_revision"],
             "verified": True, "runtime_directory": str(folder),
             "verified_weight_sha256": {n: v["sha256"] for n, v in weights.items()},
             "metadata_files": metadata, "metadata_from_pinned_public_source": True}
    write_json(output / "model_source.json", proof)
    write_json(output / "model_sources.json", {identity["official"]: str(output / "model_source.json")})
    registry = read(request["registry"])
    payload = Path(request["payload"])
    frozen, tasks = {}, []

    def bind(path):
        frozen[str(path)] = file_digest(path)

    seeds = {s for values in identity["seeds"].values() for s in values}
    if len(seeds) != 8:
        raise ValueError("Eight unique scale seeds required")
    for path in (root / "runs").rglob("run.json"):
        if read(path).get("config", {}).get("protocol", {}).get("seed") in seeds:
            raise ValueError("Fresh scale seed already used")
    for task_name in ("code", "math"):
        original = next(t for t in registry["tasks"] if t["stream"] == "qwen3-4b/" + task_name)
        manifest, final_manifest = payload / original["manifest"], payload / original["final_manifest"]
        dev_meta, _ = load_manifest(manifest)
        final_meta, final = load_final_manifest(final_manifest, manifest)
        if {r: len(v) for r, v in final.items()} != DECLARATION["scoring_items"]:
            raise ValueError("Original independent scoring sizes required")
        for path, meta in [(manifest, dev_meta), (final_manifest, final_meta)]:
            bind(path)
            for entry in meta["splits"].values():
                bind(path.parent / entry["path"])
        for seed in identity["seeds"][task_name]:
            cfg = deepcopy(read(payload / original["config"]))
            cfg["model"].update(name=identity["official"], revision=identity["revision"],
                                 chat_kwargs={"enable_thinking": False, "date_string": DECLARATION["frozen_date"]})
            cfg["protocol"]["seed"] = seed
            path = output / "configs" / (task_name + "_" + str(seed) + ".json")
            write_json(path, cfg)
            loaded = Config.load(path)
            if (loaded.training.max_steps != 50 or loaded.training.rank != 16
                    or loaded.training.learning_rate != 1e-4 or loaded.protocol.candidate_size != 2
                    or loaded.protocol.correction_attempts != 8 or loaded.protocol.correction_seed_stride != 16):
                raise ValueError("Frozen scale update and harvest budgets required")
            bind(path)
            tasks.append({"index": len(tasks), "stream": model + "/" + task_name,
                          "seed": seed, "config": str(path), "config_sha256": file_digest(path),
                          "manifest": str(manifest), "manifest_sha256": file_digest(manifest),
                          "final_manifest": str(final_manifest), "final_manifest_sha256": file_digest(final_manifest),
                          "output": str(output / "parents" / task_name / str(seed))})
    plan = {"source_tree_sha256": request["source_tree_sha256"], "declaration": DECLARATION,
            "model": model, "tasks": tasks, "frozen_files": frozen,
            "catalog_sha256": request["catalog_sha256"], "automatic_expansion": False}
    write_json(output / "plan.json", plan)
    write_json(output / "prepared.json", {"passed": True, "plan_sha256": file_digest(output / "plan.json"), "home_used_bytes": usage()})
    print(json.dumps({"scale_model_prepared": model, "parents": 8}))


def plan_for(request, model):
    output = Path(request["output"]) / model
    plan = read(output / "plan.json")
    if (plan["declaration"] != DECLARATION or plan["source_tree_sha256"] != request["source_tree_sha256"]
            or read(output / "prepared.json")["plan_sha256"] != file_digest(output / "plan.json")):
        raise ValueError("Frozen scale plan differs")
    for path, sha in plan["frozen_files"].items():
        if file_digest(path) != sha:
            raise ValueError("Frozen scale dataset/configuration differs")
    plan["plan_sha256"] = file_digest(output / "plan.json")
    return plan


def preflight(request, model):
    require_gpu_allocation()
    import torch
    from third_eye.training.hf_backend import HFBackend

    plan = plan_for(request, model)
    output = Path(request["output"]) / model / "preflight"
    output.mkdir(exist_ok=False)
    observations = []
    for task_name in ("code", "math"):
        task = next(t for t in plan["tasks"] if t["stream"].endswith("/" + task_name))
        cfg = Config.load(task["config"])
        _, dev = load_manifest(task["manifest"])
        _, final = load_final_manifest(task["final_manifest"], task["manifest"])
        backend = HFBackend(cfg)
        verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
        parent, original = backend.snapshot(), backend.state_hash()
        fixtures = [Correction(e, e.answer, 1) for e in dev["train"][:2]]
        if not all(verify_completions(verifier, [c.example for c in fixtures], [c.completion for c in fixtures])):
            raise ValueError("Native math/code gold fixtures must pass")
        training = backend.train(fixtures, cfg.protocol.seed)
        if training["optimizer_steps"] != 50 or backend.state_hash() == original:
            raise ValueError("Real 50-step adapter update failed")
        adapter = output / (task_name + "_adapter")
        backend.save_checkpoint(adapter, {"purpose": "discarded_scale_compatibility"})
        updated = backend.state_hash()
        backend.restore(parent)
        if backend.state_hash() != original or backend.load_checkpoint(adapter)["adapter_hash"] != updated:
            raise ValueError("Adapter rollback/reload failed")
        timings = {}
        for state in ("trained", "parent"):
            if state == "parent":
                backend.restore(parent)
            for role in DECLARATION["scoring_items"]:
                start = time.perf_counter()
                texts = generate_greedy(backend, final[role][:8], 512, [cfg.protocol.seed] * 8)
                if len(verify_completions(verifier, final[role][:8], texts)) != 8:
                    raise ValueError("Incomplete verifier timing")
                torch.cuda.synchronize()
                timings.setdefault(role, []).append((time.perf_counter() - start) / 8)
        # A fixed, discarded16-training-prompt harvest measures all eight
        # allowed attempts; timing alone controls reservation sufficiency.
        _, harvest = collect_corrections(backend, dev["train"][:16], verifier, cfg.protocol, cfg.protocol.seed)
        scoring = 13 * sum(max(timings[r]) * n for r, n in
                           [("target_test", 192), ("ood_test", 192), ("retention_test", 512)])
        needed = 1.5 * (scoring + 10 * len(dev["train"]) * harvest["total_seconds"] / 16
                        + 24 * training["seconds"] + 3600)
        observations.append({"task": task_name, "required_seconds": needed,
                             "reserved_seconds": request["per_parent_seconds"],
                             "discarded16_prompt_harvest_seconds": harvest["total_seconds"],
                             "per_item_final_timing": timings, "train_seconds": training["seconds"],
                             "margin": 1.5, "scored_accuracy_recorded": False,
                             "includes_three_continuations_per_candidate": True})
        backend.restore(parent)
        if backend.state_hash() != original:
            raise ValueError("Compatibility update persisted")
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output / "runtime_coverage.json", observations)
    if any(v["required_seconds"] > v["reserved_seconds"] for v in observations):
        raise ValueError("Complete estimated parent runtime exceeds seven-day reservation")
    ready = {"passed": True, "source_tree_sha256": request["source_tree_sha256"],
             "plan_sha256": plan["plan_sha256"], "model": model,
             "discarded_fixture": True, "runtime_coverage": observations}
    write_json(output / "completed.json", ready)
    write_json(output.parent / "authorized_scale.json", {
        "purpose": "frozen_empirical_scale_confirmation", "model": model,
        "declaration_sha256": digest(DECLARATION), "source_tree_sha256": request["source_tree_sha256"],
        "preflight": str(output / "completed.json"), "preflight_sha256": file_digest(output / "completed.json"),
        "tasks": [{"seed": t["seed"], "config_hash": Config.load(t["config"]).fingerprint,
                   "manifest_sha256": t["manifest_sha256"]} for t in plan["tasks"]],
        "forecasting_gate_passed_claimed": False,
    })
    print(json.dumps({"scale_preflight_passed": model, "tasks": 2}))


def collect(request, model, index):
    require_gpu_allocation()
    plan = plan_for(request, model)
    output = Path(request["output"]) / model
    if not 0 <= index < 8:
        raise ValueError("Frozen eight-parent index required")
    ready = read(output / "preflight/completed.json")
    if not ready["passed"] or ready["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Matching GPU and runtime validation required")
    headroom()
    task = plan["tasks"][index]
    run_task(request, plan, task, decision_rule=actions,
             scale_authorization=output / "authorized_scale.json")
    compact_checkpoints(task["output"])
    headroom(500_000_000)


def review(request, model):
    import numpy as np

    plan, states, hashes = plan_for(request, model), [], {}
    for task in plan["tasks"]:
        folder = Path(task["output"])
        done, decision = read(folder / "completed.json"), read(folder / "commitment.json")
        if (not done["passed"] or done["source_tree_sha256"] != request["source_tree_sha256"]
                or done["plan_sha256"] != plan["plan_sha256"]
                or done["commitment_sha256"] != file_digest(folder / "commitment.json")
                or decision["actions"] != actions(task["stream"], decision["development_h1_heads"])):
            raise ValueError("Complete immutable scale parent required")
        retention = read(folder / "lossless_retention.json")
        if not retention["passed"]:
            raise ValueError("Lossless adapter retention required")
        for item in retention["records"]:
            if file_digest(folder / item["archive"]) != item["archive_sha256"]:
                raise ValueError("Archived adapter changed")
        rows, _ = inspect_task({**task, "output": str(folder / "collection")},
                               [50_000_000, 100_000_000], request["source_tree_sha256"])
        _, final = load_final_manifest(task["final_manifest"], task["manifest"])
        artifacts = {}
        for name, _, adapter_hash in checkpoints(rows):
            dest = folder / "scoring" / name
            for filename, key in [("scores.json", "scores_sha256"), ("items.jsonl", "items_sha256")]:
                if file_digest(dest / filename) != done["scores"][name][key]:
                    raise ValueError("Immutable scale scoring changed")
                hashes[str(dest / filename)] = done["scores"][name][key]
            scored = read(dest / "scores.json")
            if scored["source_tree_sha256"] != request["source_tree_sha256"] or scored["plan_sha256"] != plan["plan_sha256"]:
                raise ValueError("Scoring source/plan differs")
            artifacts[name] = read_scored_items(dest, final, adapter_hash, done["commitment_sha256"])
        for k, row in enumerate(rows):
            for d, obs in enumerate(row["continuation_replicates"]):
                if not obs["continuation_available"] and any(not np.array_equal(artifacts[f"m2_{k}_{d}"][r], artifacts[f"m1_{k}"][r]) for r in final):
                    raise ValueError("Terminal continuation changed receiver")
        states.append({"stream": task["stream"], "seed": task["seed"], "actions": decision["actions"],
                       "artifacts": artifacts, "ids": {r: [(e.id, e.prompt_hash) for e in xs] for r, xs in final.items()}})
    output = Path(request["output"]) / model / "evidence"
    summarize(states, model, output, {"source_tree_sha256": request["source_tree_sha256"],
                                     "plan_sha256": plan["plan_sha256"], "input_hashes": hashes})
    write_json(output / "completed.json", {"passed": True, "parents": 8,
                                           "report_sha256": file_digest(output / "report.json")})


def finalize(request):
    reports = {}
    for model in MODELS:
        output = Path(request["output"]) / model / "evidence"
        done = read(output / "completed.json")
        if not done["passed"] or file_digest(output / "report.json") != done["report_sha256"]:
            raise ValueError("Both complete scale cohorts required")
        reports[model] = {"report": str(output / "report.json"), **done}
    write_json(Path(request["output"]) / "completed.json", {
        "passed": True, "models": reports, "parents": 16, "scored_checkpoints": 208,
        "Vista_SUs": 0, "automatic_expansion": False,
        "primary_family": "two model-specific selected_change intervals at97.5% confidence",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["rotate", "prepare", "preflight", "run", "review", "finalize"])
    parser.add_argument("--request", required=True)
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    request = checked(args.request)
    if args.command in {"rotate", "finalize"}:
        {"rotate": rotate, "finalize": finalize}[args.command](request)
    elif args.command == "run":
        collect(request, args.model, args.index)
    else:
        {"prepare": prepare, "preflight": preflight, "review": review}[args.command](request, args.model)


if __name__ == "__main__":
    main()
