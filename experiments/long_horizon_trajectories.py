"""Paired fixed-recipe T=5 trajectories, with T=3 and T=5 scoring."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config, ModelConfig, TrainingConfig, ProtocolConfig
from third_eye.data.schema import Correction
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import evaluate_final, load_final_manifest
from third_eye.evaluation.runner import verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.confirmation import read_scored_items
from third_eye.experiments.long_horizon import (
    DECLARATION, MODELS, POLICIES, ROLES, configured, item_contrasts, step_seeds, summarize, train_step,
)
from third_eye.experiments.regime_revision import unique_batch
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.updates.corrections import collect_corrections
from next_round_regimes import eligible_pool

SCORED = ("m0", "m1", "native_t2", "mitigated_t2", "native_t3", "mitigated_t3", "native_t5", "mitigated_t5")


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, validation=True):
    require_compute_allocation()
    request = read(path)
    if request["declaration"] != DECLARATION or request["source_tree_sha256"] != source_inventory()["source_tree_sha256"]:
        raise ValueError("Frozen trajectory source/declaration differs")
    for field in ("template", "manifest", "final_manifest", "primary_request", "model_sources"):
        if file_digest(request[field]) != request[field+"_sha256"]:
            raise ValueError("Frozen trajectory input differs: "+field)
    if validation:
        valid = read(request["validation"])
        if (not valid["passed"] or valid["request_sha256"] != file_digest(path) or
                valid["source_tree_sha256"] != request["source_tree_sha256"]):
            raise ValueError("Matching complete CPU validation required")
    return request


def cohort_config(request, model, index):
    return configured(Config.load(request["template"]), model, MODELS[model]["seeds"][index])


def compatibility(request, model):
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend
    cfg = cohort_config(request, model, 0)
    folder = Path(request["output"])/"compatibility"/model
    folder.mkdir(parents=True, exist_ok=False)
    _, selection = load_manifest(request["manifest"])
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    before, snapshot = backend.state_hash(), backend.snapshot()
    fixtures = [Correction(e, e.answer, 1) for e in selection["train"][:8]]
    if not all(verify_completions(verifier, [c.example for c in fixtures], [c.completion for c in fixtures])):
        raise ValueError("Training-only fixtures fail verification")
    fit = backend.train(fixtures, cfg.protocol.seed+1, max_steps=1)
    backend.restore(snapshot)
    if backend.state_hash() != before:
        raise ValueError("Real GPU fixture did not roll back exactly")
    # Test actual generation/choice-scoring paths without looking at final items.
    predictions = backend.generate_many([e.prompt for e in selection["train"][:32]], 16, [0]*32)
    if len(predictions) != 32:
        raise ValueError("Incomplete batched generation compatibility result")
    choices = [e for e in selection["retention_dev"] if e.metadata.get("verifier") == "multiple_choice"][:4]
    if len(backend.predict_choices([e.prompt for e in choices])) != len(choices):
        raise ValueError("Incomplete conditional-likelihood compatibility result")
    if backend.state_hash() != before:
        raise ValueError("Generation compatibility changed the checkpoint")
    write_json(folder/"completed.json", {"passed": True, "model": model, "training": fit,
        "exact_rollback": True, "research_update": False, "generation_batch": 32,
        "source_tree_sha256": request["source_tree_sha256"], "config": cfg.to_dict(),
        "request_sha256": file_digest(Path(request["output"])/"request.json")})


def copy_checkpoint(source, dest):
    """Read archived source adapters; decompress into this account only."""
    source, dest = Path(source), Path(dest)
    state = read(source/"state.json")
    dest.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and (path.suffix in {".json", ".jinja", ".txt"}):
            shutil.copyfile(path, dest/path.name)
    for name, expected in state["weights"].items():
        raw = source/name
        if raw.exists():
            shutil.copyfile(raw, dest/name)
        else:
            with gzip.open(raw.with_suffix(raw.suffix+".gz"), "rb") as src, (dest/name).open("xb") as dst:
                shutil.copyfileobj(src, dst, 8*1024**2)
        if file_digest(dest/name) != expected:
            raise ValueError("Imported archived adapter failed byte verification")
    return state


def saved_config(value):
    training, protocol = dict(value["training"]), dict(value["protocol"])
    training["target_modules"] = tuple(training["target_modules"])
    protocol["utility_weights"] = tuple(protocol["utility_weights"])
    return Config(ModelConfig(**value["model"]), TrainingConfig(**training), ProtocolConfig(**protocol))


def run(request, model, index):
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend
    cfg = cohort_config(request, model, index)
    seed = cfg.protocol.seed
    root = Path(request["output"])
    compatible = read(root/"compatibility"/model/"completed.json")
    if not compatible["passed"] or compatible["source_tree_sha256"] != request["source_tree_sha256"]:
        raise ValueError("Matching real model/CUDA compatibility required")
    folder = root/"parents"/model/str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _, selection = load_manifest(request["manifest"])
    _, final = load_final_manifest(request["final_manifest"], request["manifest"])
    if len(selection["train"]) != 1024 or {r: len(xs) for r, xs in final.items()} != DECLARATION["evaluation_items"]:
        raise ValueError("Frozen training/evaluation counts differ")
    write_json(folder/"design.json", {"declaration": DECLARATION, "model": model, "seed": seed,
        "config": cfg.to_dict(), "request_sha256": file_digest(root/"request.json"),
        "future_outcomes_used_for_decisions": False})
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    backend = HFBackend(cfg)
    snapshots, states, inherited = {}, {}, {}
    primary_binding = None

    def save(name, metadata):
        backend.save_checkpoint(folder/"checkpoints"/name, metadata)
        snapshots[name] = backend.snapshot()
        states[name] = {**metadata, "adapter_hash": backend.state_hash(),
            "state_sha256": file_digest(folder/"checkpoints"/name/"state.json")}
        write_json(folder/"progress.json", {"phase": "updates", "states": states,
            "elapsed_seconds": time.perf_counter()-started, "final_scoring_started": False})

    def harvest(generation, policy, terminal=False):
        hseed, bseed, oseed = step_seeds(seed, generation)
        if terminal:
            pool, rejected, batch = [], [], None
            stats = {"not_harvested": "absorbing terminal state"}
        else:
            pool, stats = collect_corrections(backend, selection["train"], verifier, cfg.protocol, hseed)
            pool, rejected = eligible_pool(backend, pool)
            batch = unique_batch(pool, 8, bseed)
        path = folder/"donors"/(policy+"_t"+str(generation)+".json")
        write_json(path, {"generation": generation, "policy": policy, "harvest_seed": hseed,
            "stats": stats, "sft_length_rejections": rejected, "corrections": [c.to_dict() for c in pool],
            "batch": [c.to_dict() for c in batch] if batch is not None else [],
            "batch_sha256": digest([c.to_dict() for c in batch] if batch is not None else [])})
        return batch, oseed, {"donor": str(path.relative_to(folder)), "donor_sha256": file_digest(path)}

    if model == "qwen3-4b":
        from positive_regime_confirmation import inspect_parent
        primary = read(request["primary_request"])
        validation = read(primary["validation"])
        if (not validation["passed"] or validation["request_sha256"] != request["primary_request_sha256"] or
                source_inventory(primary["source"])["source_tree_sha256"] != primary["source_tree_sha256"]):
            raise ValueError("Original primary provenance differs")
        inspect_parent(primary, index, final)
        source = Path(primary["output"])/"parents"/str(seed)
        done, commit = read(source/"completed.json"), read(source/"commitment.json")
        if not read(source/"lossless_retention.json")["passed"]:
            raise ValueError("Complete retained primary parent required")
        primary_binding = {"parent": str(source), "completed_sha256": file_digest(source/"completed.json"),
            "commitment_sha256": done["commitment_sha256"], "canonical_draw": 0,
            "request_sha256": request["primary_request_sha256"]}
        for name, old in (("m0", "m0"), ("m1", "m1"), ("native_t2", "d0_native"),
                          ("mitigated_t2", "d0_mitigated")):
            saved = copy_checkpoint(source/"checkpoints"/old, folder/"checkpoints"/name)
            backend.config = saved_config(saved["config"])
            backend.load_checkpoint(folder/"checkpoints"/name)
            backend.config = cfg
            snapshots[name] = backend.snapshot()
            terminal = not done["initial_available"] or (name.endswith("t2") and not done["continuation_available"][0])
            states[name] = {"adapter_hash": backend.state_hash(), "inherited": True,
                "terminal": terminal, "generation": 0 if name == "m0" else 1 if name == "m1" else 2,
                "state_sha256": file_digest(folder/"checkpoints"/name/"state.json"),
                "source_state_sha256": commit["states"][old]["state_sha256"]}
            dest = folder/"scoring"/name
            dest.mkdir(parents=True, exist_ok=False)
            for filename in ("scores.json", "items.jsonl"):
                shutil.copyfile(source/"scoring"/old/filename, dest/filename)
            inherited[name] = done["scores"][old]
        first_generation = 3
    else:
        save("m0", {"generation": 0, "terminal": False})
        batch, oseed, donor = harvest(1, "shared")
        record = train_step(backend, cfg, "native", 1, batch, oseed)
        save("m1", {**record, **donor})
        m1 = snapshots["m1"]
        batch, oseed, donor = harvest(2, "shared", record["terminal"])
        for policy in POLICIES:
            backend.restore(m1)
            record = train_step(backend, cfg, policy, 2, batch, oseed, states["m1"]["terminal"])
            save(policy+"_t2", {**record, **donor})
        first_generation = 3
    for generation in range(first_generation, 6):
        for policy in POLICIES:
            previous = policy+"_t"+str(generation-1)
            backend.restore(snapshots[previous])
            terminal = states[previous]["terminal"]
            batch, oseed, donor = harvest(generation, policy, terminal)
            record = train_step(backend, cfg, policy, generation, batch, oseed, terminal)
            save(policy+"_t"+str(generation), {**record, **donor})
    # The entire trajectory is fixed before any new final score is computed.
    write_json(folder/"commitment.json", {"states": states, "design_sha256": file_digest(folder/"design.json"),
        "primary_binding": primary_binding, "inherited_scores": inherited,
        "source_tree_sha256": request["source_tree_sha256"], "forecaster_fits": 0})
    commitment_hash = file_digest(folder/"commitment.json")
    scores, seen = {}, {}
    backend.config = cfg
    for name in SCORED:
        stamp = time.perf_counter()
        state, dest = states[name], folder/"scoring"/name
        backend.restore(snapshots[name])
        if backend.state_hash() != state["adapter_hash"]:
            raise ValueError("Committed trajectory state changed before scoring")
        metadata = {"commitment_sha256": commitment_hash, "checkpoint": name, "parent_seed": seed,
            "model": model, "request_sha256": file_digest(root/"request.json")}
        if name in inherited:
            pass  # Byte-identical original scores retain their original commitment.
        elif state["adapter_hash"] in seen:
            prior = folder/"scoring"/seen[state["adapter_hash"]]
            dest.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(prior/"items.jsonl", dest/"items.jsonl")
            write_json(dest/"scores.json", {**read(prior/"scores.json"), **metadata,
                "identity_reuse": str(prior), "identity_reuse_items_sha256": file_digest(prior/"items.jsonl")})
        else:
            evaluate_final(backend, final, verifier, dest, 512, 0, metadata)
        seen.setdefault(state["adapter_hash"], name)
        scores[name] = {"scores_sha256": file_digest(dest/"scores.json"),
            "items_sha256": file_digest(dest/"items.jsonl"), "seconds": time.perf_counter()-stamp}
        write_json(folder/"progress.json", {"phase": "scoring", "scores": scores,
            "elapsed_seconds": time.perf_counter()-started})
    write_json(folder/"completed.json", {"passed": True, "model": model, "seed": seed,
        "request_sha256": file_digest(root/"request.json"), "commitment_sha256": commitment_hash,
        "source_tree_sha256": request["source_tree_sha256"], "scores": scores,
        "elapsed_seconds": time.perf_counter()-started, "forecaster_fits": 0})


def inspect(request, model, index):
    seed = MODELS[model]["seeds"][index]
    folder = Path(request["output"])/"parents"/model/str(seed)
    done, commitment = read(folder/"completed.json"), read(folder/"commitment.json")
    if (not done["passed"] or done["seed"] != seed or done["model"] != model or
            done["source_tree_sha256"] != request["source_tree_sha256"] or
            done["request_sha256"] != file_digest(Path(request["output"])/"request.json") or
            done["commitment_sha256"] != file_digest(folder/"commitment.json") or
            commitment["design_sha256"] != file_digest(folder/"design.json") or set(done["scores"]) != set(SCORED)):
        raise ValueError("Complete fixed trajectory provenance required")
    _, final = load_final_manifest(request["final_manifest"], request["manifest"])
    if set(commitment["states"]) != {"m0", "m1"} | {p+"_t"+str(t) for p in POLICIES for t in range(2, 6)}:
        raise ValueError("All ten cumulative trajectory states required")
    for name, state in commitment["states"].items():
        path = folder/"checkpoints"/name/"state.json"
        if file_digest(path) != state["state_sha256"] or read(path)["adapter_hash"] != state["adapter_hash"]:
            raise ValueError("Trajectory checkpoint identity changed")
        if "donor" in state:
            donor_path = folder/state["donor"]
            donor = read(donor_path)
            if file_digest(donor_path) != state["donor_sha256"] or digest(donor["batch"]) != donor["batch_sha256"]:
                raise ValueError("Trajectory donor data changed")
            if [c["example"]["id"] for c in donor["batch"]] != state["batch_ids"]:
                raise ValueError("Update does not match its sealed donor batch")
            generation, policy = state["generation"], state["policy"]
            previous = "m0" if generation == 1 else "m1" if generation == 2 else policy+"_t"+str(generation-1)
            if state["start_hash"] != commitment["states"][previous]["adapter_hash"]:
                raise ValueError("Broken cumulative checkpoint chain")
            from third_eye.experiments.long_horizon import learning_rate
            if (state["optimizer_seed"] != step_seeds(seed, generation)[2] or
                    state["learning_rate"] != learning_rate(policy, generation) or
                    (commitment["states"][previous]["terminal"] and not state["terminal"])):
                raise ValueError("Trajectory seed, LR or absorbing-terminal rule differs")
            if state["terminal"]:
                if state["adapter_hash"] != state["start_hash"] or state["training"] is not None:
                    raise ValueError("Terminal trajectory identity failed")
            elif (state["training"]["optimizer_steps"] != 25 or len(state["batch_ids"]) != 8 or
                    len(set(state["batch_ids"])) != 8):
                raise ValueError("Unique batch or optimizer budget differs")
    observed = {}
    for name, hashes in done["scores"].items():
        dest = folder/"scoring"/name
        if any(file_digest(dest/file) != hashes[key] for file, key in
               (("scores.json", "scores_sha256"), ("items.jsonl", "items_sha256"))):
            raise ValueError("Scoring outcome changed")
        own = name not in commitment["inherited_scores"]
        if not own and any(hashes[k] != commitment["inherited_scores"][name][k] for k in
                           ("scores_sha256", "items_sha256")):
            raise ValueError("Inherited outcomes changed")
        expected = done["commitment_sha256"] if own else commitment["primary_binding"]["commitment_sha256"]
        observed[name] = read_scored_items(dest, final, commitment["states"][name]["adapter_hash"], expected)
    return {r: item_contrasts({name: values[r] for name, values in observed.items()}) for r in ROLES}


def review(request, model):
    import numpy as np
    matrices = {r: [] for r in ROLES}
    records = []
    for index, seed in enumerate(MODELS[model]["seeds"]):
        folder = Path(request["output"])/"parents"/model/str(seed)
        if not read(folder/"lossless_retention.json")["passed"]:
            raise ValueError("Lossless adapter retention required")
        values = inspect(request, model, index)
        for r in ROLES:
            matrices[r].append(values[r])
        records.append(read(folder/"completed.json"))
    result = summarize({r: np.stack(xs) for r, xs in matrices.items()}, model)
    write_json(Path(request["output"])/(model+"_review.json"), {"passed": True, "statistics": result,
        "elapsed_parent_seconds": [r["elapsed_seconds"] for r in records], "declaration": DECLARATION,
        "request_sha256": file_digest(Path(request["output"])/"request.json"), "automatic_expansion": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compatibility", "run", "archive", "review"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--index", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    request = checked(args.request)
    os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
    if args.mode == "compatibility":
        compatibility(request, args.model)
    elif args.mode == "run":
        run(request, args.model, args.index)
    elif args.mode == "review":
        review(request, args.model)
    else:
        from third_eye.experiments.scale import compact_checkpoints
        inspect(request, args.model, args.index)
        compact_checkpoints(Path(request["output"])/"parents"/args.model/str(MODELS[args.model]["seeds"][args.index]))


if __name__ == "__main__":
    main()
