"""Fresh, fixed-recipe confirmation with matched native/direct continuations."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.config import Config
from third_eye.data.schema import Correction
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import load_final_manifest, evaluate_final
from third_eye.evaluation.runner import verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.experiments.confirmation import read_scored_items
from third_eye.experiments.positive_regime import (
    DECLARATION, ARMS, ROLES, arm_plan, item_contrasts, summarize,
)
from third_eye.experiments.regime_revision import unique_batch, select_regime
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.updates.corrections import collect_corrections
from next_round_regimes import eligible_pool


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, validation=True):
    require_compute_allocation()
    request = read(path)
    if (request["declaration"] != DECLARATION or request["source_tree_sha256"] !=
            source_inventory()["source_tree_sha256"]):
        raise ValueError("Frozen confirmation source/declaration changed")
    for field in ("config", "manifest", "final_manifest", "development_review", "model_sources"):
        if file_digest(request[field]) != request[field+"_sha256"]:
            raise ValueError("Frozen input differs: "+field)
    review = read(request["development_review"])
    choice = select_regime(review["records"])
    if (not review["passed"] or review["final_outcomes_generated"] or
            choice != review["eligible_regime"] or choice is None or
            choice["model"] != DECLARATION["model"] or choice["recipe"] != DECLARATION["recipe"]):
        raise ValueError("Development-only recipe decision required")
    if validation:
        validated = read(request["validation"])
        if (not validated["passed"] or validated["request_sha256"] != file_digest(path) or
                validated["source_tree_sha256"] != request["source_tree_sha256"]):
            raise ValueError("Matching complete compute-node validation required")
    return request


def configured(request, seed):
    cfg = Config.load(request["config"])
    cfg = replace(cfg, training=replace(cfg.training, learning_rate=DECLARATION["learning_rate"],
        max_steps=DECLARATION["steps"]), protocol=replace(cfg.protocol, seed=seed, depth=1,
        candidate_size=DECLARATION["batch_size"]))
    if (cfg.model.name != "Qwen/Qwen3-4B" or cfg.model.revision !=
            "1cfa9a7208912126459214e8b04321603b3df60c" or
            cfg.model.chat_kwargs != {"enable_thinking": False, "date_string": "11 Oct 2026"} or
            cfg.training.rank != 16 or cfg.training.alpha != 32 or
            cfg.training.max_sequence_length != 1024 or cfg.protocol.correction_attempts != 8 or
            cfg.protocol.max_new_tokens != 512 or cfg.protocol.generation_batch_size != 32 or
            cfg.protocol.sampled_batch_size != 16 or cfg.protocol.multiple_choice_scoring !=
            "conditional_likelihood" or not cfg.protocol.deterministic_execution):
        raise ValueError("Positive-regime configuration drift")
    return cfg


def paired_updates(backend, cfg, parent, m1, parent_hash, m1_hash, batch, seed, save):
    """Restore before each arm; every train call creates a fresh optimizer."""
    plan = arm_plan(parent_hash, m1_hash, batch, seed)
    for arm, spec in plan.items():
        backend.config = replace(cfg, training=replace(cfg.training, learning_rate=spec["learning_rate"]))
        backend.restore(parent if arm == "direct" else m1)
        if backend.state_hash() != spec["start_hash"]:
            raise ValueError("Matched arm did not start at committed state")
        training = backend.train(batch, seed, max_steps=spec["steps"]) if batch is not None else None
        if training is not None and training["optimizer_steps"] != DECLARATION["steps"]:
            raise ValueError("Unexpected optimizer budget")
        if batch is None and backend.state_hash() != spec["start_hash"]:
            raise ValueError("Terminal arm must preserve its starting state exactly")
        save(arm, {**spec, "training": training, "adapter_hash": backend.state_hash()})
    backend.config = cfg
    return plan


def run_parent(request, index):
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend
    seed = DECLARATION["seeds"][index]
    cfg = configured(request, seed)
    folder = Path(request["output"])/"parents"/str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _, selection = load_manifest(request["manifest"])
    if len(selection["train"]) != DECLARATION["training_prompts"]:
        raise ValueError("Frozen correction-harvest budget changed")
    # Inspect final identities now, never outcomes: no recipe choice is made here.
    _, final = load_final_manifest(request["final_manifest"], request["manifest"])
    if {r: len(xs) for r, xs in final.items()} != DECLARATION["evaluation_items"]:
        raise ValueError("Frozen final counts differ")
    write_json(folder/"design.json", {"declaration": DECLARATION, "seed": seed,
        "request_sha256": file_digest(Path(request["output"])/"request.json"),
        "config": cfg.to_dict(), "config_sha256": cfg.fingerprint,
        "selection_uses_final_outcomes": False})
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    parent = backend.snapshot()
    parent_hash = backend.state_hash()
    fixtures = [Correction(e, e.answer, 1) for e in selection["train"][:8]]
    if not all(verify_completions(verifier, [c.example for c in fixtures], [c.completion for c in fixtures])):
        raise ValueError("Training-only compatibility fixtures fail verification")
    compatibility = backend.train(fixtures, seed+1, max_steps=1)
    backend.restore(parent)
    if backend.state_hash() != parent_hash:
        raise ValueError("GPU adapter compatibility test failed exact rollback")
    write_json(folder/"compatibility.json", {"passed": True, "training": compatibility,
        "exact_rollback": True, "research_update": False})
    snapshots, states = {}, {}

    def save(name, metadata):
        backend.save_checkpoint(folder/"checkpoints"/name, metadata)
        snapshots[name] = backend.snapshot()
        states[name] = {**metadata, "adapter_hash": backend.state_hash(),
            "state_sha256": file_digest(folder/"checkpoints"/name/"state.json")}
        write_json(folder/"progress.json", {"states": states, "elapsed_seconds": time.perf_counter()-started,
            "phase": "training", "final_scoring_started": False})

    save("m0", {"seed": seed, "initial_available": True})
    pool, stats = collect_corrections(backend, selection["train"], verifier, cfg.protocol, seed)
    pool, rejected = eligible_pool(backend, pool)
    batch = unique_batch(pool, 8, seed+17)
    write_json(folder/"initial_pool.json", {"stats": stats, "sft_length_rejections": rejected,
        "corrections": [c.to_dict() for c in pool],
        "batch": [c.to_dict() for c in batch] if batch is not None else []})
    training = backend.train(batch, seed+101, max_steps=25) if batch is not None else None
    if training is not None and training["optimizer_steps"] != 25:
        raise ValueError("First update optimizer budget differs")
    initial_available = batch is not None
    save("m1", {"seed": seed, "initial_available": initial_available, "training": training,
        "optimizer_seed": seed+101, "batch_sha256": digest([c.to_dict() for c in batch] if batch else []),
        "start_hash": parent_hash})
    m1, m1_hash = snapshots["m1"], states["m1"]["adapter_hash"]
    if not initial_available and m1_hash != parent_hash:
        raise ValueError("Initial correction scarcity must preserve M1=M0")
    donors = []
    for draw in range(3):
        draw_seed = seed+10_000+draw*100_000
        backend.restore(m1)
        if initial_available:
            future, future_stats = collect_corrections(
                backend, selection["train"], verifier, cfg.protocol, draw_seed)
            future, rejected = eligible_pool(backend, future)
            donor = unique_batch(future, 8, draw_seed+19)
        else:
            future, rejected, donor = [], [], None
            future_stats = {"not_harvested": "initial batch unavailable; frozen identity parent"}
        donor_record = {"draw": draw, "harvest_seed": draw_seed, "harvest": future_stats,
            "sft_length_rejections": rejected, "corrections": [c.to_dict() for c in future],
            "batch": [c.to_dict() for c in donor] if donor is not None else [],
            "continuation_available": int(donor is not None)}
        donor_record["batch_sha256"] = digest(donor_record["batch"])
        donors.append(donor_record)
        write_json(folder/"donors"/(str(draw)+".json"), donor_record)

        def save_arm(arm, metadata):
            save(f"d{draw}_{arm}", {**metadata, "draw": draw, "arm": arm,
                "batch_sha256": donor_record["batch_sha256"]})

        paired_updates(backend, cfg, parent, m1, parent_hash, m1_hash, donor, draw_seed+31, save_arm)
    # All updates and donors are sealed before any final benchmark scoring.
    commitment = {"design_sha256": file_digest(folder/"design.json"), "states": states,
        "donor_sha256": [file_digest(folder/"donors"/(str(d)+".json")) for d in range(3)],
        "source_tree_sha256": request["source_tree_sha256"],
        "final_manifest_sha256": request["final_manifest_sha256"], "forecaster_fits": 0}
    write_json(folder/"commitment.json", commitment)
    commitment_hash = file_digest(folder/"commitment.json")
    scores, seen = {}, {}
    for name, state in states.items():
        backend.restore(snapshots[name])
        if backend.state_hash() != state["adapter_hash"]:
            raise ValueError("Sealed checkpoint differs before final scoring")
        dest = folder/"scoring"/name
        metadata = {"commitment_sha256": commitment_hash, "checkpoint": name, "seed": seed,
                    "request_sha256": file_digest(Path(request["output"])/"request.json")}
        stamp = time.perf_counter()
        if state["adapter_hash"] in seen:
            prior = folder/"scoring"/seen[state["adapter_hash"]]
            dest.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(prior/"items.jsonl", dest/"items.jsonl")
            write_json(dest/"scores.json", {**read(prior/"scores.json"), **metadata,
                "identity_reuse": str(prior), "identity_reuse_items_sha256": file_digest(prior/"items.jsonl")})
        else:
            evaluate_final(backend, final, verifier, dest, 512, 0, metadata)
            seen[state["adapter_hash"]] = name
        scores[name] = {"scores_sha256": file_digest(dest/"scores.json"),
            "items_sha256": file_digest(dest/"items.jsonl"), "seconds": time.perf_counter()-stamp}
        write_json(folder/"progress.json", {"phase": "final_scoring", "scores": scores,
            "elapsed_seconds": time.perf_counter()-started, "final_scoring_started": True})
    write_json(folder/"completed.json", {"passed": True, "seed": seed, "initial_available": initial_available,
        "continuation_available": [d["continuation_available"] for d in donors],
        "elapsed_seconds": time.perf_counter()-started, "commitment_sha256": commitment_hash,
        "request_sha256": file_digest(Path(request["output"])/"request.json"),
        "source_tree_sha256": request["source_tree_sha256"], "scores": scores, "forecaster_fits": 0})


def inspect_parent(request, index, final):
    """Independent saved-artifact checks, used before archival and final analysis."""
    import numpy as np
    seed = DECLARATION["seeds"][index]
    folder = Path(request["output"])/"parents"/str(seed)
    done = read(folder/"completed.json")
    commitment = read(folder/"commitment.json")
    if (not done["passed"] or done["seed"] != seed or
            done["request_sha256"] != file_digest(Path(request["output"])/"request.json") or
            done["source_tree_sha256"] != request["source_tree_sha256"] or
            file_digest(folder/"commitment.json") != done["commitment_sha256"] or
            file_digest(folder/"design.json") != commitment["design_sha256"]):
        raise ValueError("Parent provenance differs")
    if set(done["scores"]) != {"m0", "m1"} | {f"d{d}_{a}" for d in range(3) for a in ARMS}:
        raise ValueError("All eleven checkpoints required")
    observed = {}
    for name, hashes in done["scores"].items():
        state = commitment["states"][name]
        state_path = folder/"checkpoints"/name/"state.json"
        if (file_digest(state_path) != state["state_sha256"] or
                read(state_path)["adapter_hash"] != state["adapter_hash"]):
            raise ValueError("Checkpoint state differs")
        score = folder/"scoring"/name
        if any(file_digest(score/filename) != hashes[field] for filename, field in
               (("scores.json", "scores_sha256"), ("items.jsonl", "items_sha256"))):
            raise ValueError("Scoring file changed")
        observed[name] = read_scored_items(score, final, state["adapter_hash"], done["commitment_sha256"])
    m0_hash = commitment["states"]["m0"]["adapter_hash"]
    m1_hash = commitment["states"]["m1"]["adapter_hash"]
    if not done["initial_available"] and m0_hash != m1_hash:
        raise ValueError("Initial terminal identity differs")
    for d in range(3):
        donor_path = folder/"donors"/(str(d)+".json")
        donor = read(donor_path)
        if (file_digest(donor_path) != commitment["donor_sha256"][d] or
                digest(donor["batch"]) != donor["batch_sha256"] or
                donor["continuation_available"] != done["continuation_available"][d]):
            raise ValueError("Donor commitment differs")
        batch = donor["batch"]
        if batch and (len(batch) != 8 or len({c["example"]["id"] for c in batch}) != 8 or
                      any(c["example"]["split"] != "train" for c in batch)):
            raise ValueError("Invalid donor identities")
        for arm in ARMS:
            state = commitment["states"][f"d{d}_{arm}"]
            start = m0_hash if arm == "direct" else m1_hash
            if (state["start_hash"] != start or state["optimizer_seed"] != seed+10_000+d*100_000+31 or
                    state["batch_sha256"] != donor["batch_sha256"] or
                    state["batch_ids"] != [c["example"]["id"] for c in batch] or
                    state["learning_rate"] != DECLARATION["mitigation_learning_rate" if arm ==
                                                         "mitigated" else "learning_rate"]):
                raise ValueError("Matched optimizer/donor budget differs")
            if batch:
                if state["training"]["optimizer_steps"] != 25 or state["steps"] != 25:
                    raise ValueError("Optimizer steps differ")
            elif state["adapter_hash"] != start or state["training"] is not None or state["steps"] != 0:
                raise ValueError("Terminal continuation identity differs")
    return {r: item_contrasts(observed["m0"][r], observed["m1"][r], np.asarray([
        [observed[f"d{d}_{a}"][r] for a in ARMS] for d in range(3)])) for r in ROLES}


def review(request):
    import numpy as np
    _, final = load_final_manifest(request["final_manifest"], request["manifest"])
    matrices, parents = {r: [] for r in ROLES}, []
    for i, seed in enumerate(DECLARATION["seeds"]):
        folder = Path(request["output"])/"parents"/str(seed)
        if not read(folder/"lossless_retention.json")["passed"]:
            raise ValueError("Verified retained adapters required")
        values = inspect_parent(request, i, final)
        for r in ROLES:
            matrices[r].append(values[r])
        parents.append(read(folder/"completed.json"))
    result = summarize({r: np.stack(xs) for r, xs in matrices.items()},
                       DECLARATION["bootstrap_replicates"], DECLARATION["bootstrap_seed"])
    write_json(Path(request["output"])/"confirmation_review.json", {
        "passed": True, "declaration": DECLARATION, "statistics": result,
        "request_sha256": file_digest(Path(request["output"])/"request.json"),
        "initial_scarce_parents": sum(not p["initial_available"] for p in parents),
        "terminal_draws": sum(3-sum(p["continuation_available"]) for p in parents),
        "elapsed_parent_seconds": [p["elapsed_seconds"] for p in parents],
        "automatic_expansion": False, "forecaster_fits": 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("parent", "review"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--index", type=int, choices=range(20))
    args = parser.parse_args()
    request = checked(args.request)
    os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
    if args.mode == "parent":
        if args.index is None:
            parser.error("parent requires --index")
        run_parent(request, args.index)
    else:
        review(request)


if __name__ == "__main__":
    main()
