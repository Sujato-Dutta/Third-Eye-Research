"""Time paired overlap updates on archived inputs and development items only."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from revision_overlap_batches import corrections
from revision_paired_probe import (
    ROLES, SMOKE_LAMBDA, checked_inputs, probe, read, score,
)
from third_eye.cluster import require_compute_allocation, require_gpu_allocation
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.evaluation.verifiers import VerifierRegistry
from third_eye.io import file_digest, write_json

DOSES = ("low", "medium", "high")
DEV_ITEMS = 24


def validate_design(args):
    splits, _ = checked_inputs(args)
    seed = int(args.parent.name)
    design = read(args.feasibility)
    eligible = [row for row in design["records"] if row["seed"] == seed and row["available"]]
    if not design["passed"] or not design["shared_doses"] or not eligible:
        raise ValueError("Validated batch matching required for this parent")
    row = min(eligible, key=lambda r: r["draw"])
    pool = corrections(read(args.parent / "donors" / f'{row["draw"]}.json')["corrections"])
    by_id = {c.example.id: c for c in pool}
    initial = corrections(read(args.parent / "initial_pool.json")["batch"])
    batches = {d: [by_id[k] for k in row["batches"][d]["batch_ids"]] for d in DOSES}
    if any(len(batch) != 8 or len({c.example.id for c in batch}) != 8 for batch in batches.values()):
        raise ValueError("Invalid archived dose batches")
    pooled = initial + batches["medium"]
    if len(pooled) != 16 or len({c.example.id for c in pooled}) != 16:
        raise ValueError("Pooled control requires distinct first and next problems")
    return splits, seed, row, batches, pooled


def run(args):
    from long_horizon_trajectories import copy_checkpoint, saved_config
    import torch
    from third_eye.training.hf_backend import HFBackend

    require_gpu_allocation()
    smoke = read(args.smoke)
    if not smoke["passed"] or not smoke["exact_rollback"] or smoke["final_eval_read"]:
        raise ValueError("Successful paired GPU compatibility check required")
    if torch.cuda.get_device_properties(0).total_memory < 38_000_000_000:
        raise ValueError("Timed pilot requires a full 40 GB class GPU")
    if args.home_cap is not None:
        usage = int(subprocess.check_output(["du", "-sx", "-B1", str(Path.home())],
                                            text=True).split()[0])
        if usage + 2_000_000_000 > args.home_cap:
            raise ValueError("Home no longer has the required operating reserve")
    if shutil.disk_usage(tempfile.gettempdir()).free < 2_000_000_000:
        raise ValueError("GPU node has insufficient temporary space")
    splits, seed, row, batches, pooled = validate_design(args)
    stamp = time.perf_counter()
    state = read(args.parent / "checkpoints" / "m0" / "state.json")
    if state["model_source"] is not None:
        raise ValueError("Pilot expects the verified official-cache base checkpoint")
    cfg = saved_config(state["config"])
    backend = HFBackend(cfg)
    snapshots, hashes = {}, {}
    for name in ("m0", "m1"):
        with tempfile.TemporaryDirectory(prefix="third-eye-overlap-") as temp:
            path = Path(temp) / "adapter"
            saved = copy_checkpoint(args.parent / "checkpoints" / name, path)
            backend.config = saved_config(saved["config"])
            backend.load_checkpoint(path)
        snapshots[name], hashes[name] = backend.snapshot(), backend.state_hash()
    backend.config = cfg
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    dev = {}
    for role in ROLES:
        complete = []
        for item in splits[role]:
            try:
                backend._encode(item.prompt, item.answer)
            except ValueError as error:
                if "SFT example exceeds" not in str(error):
                    raise
                continue
            complete.append(item)
            if len(complete) == 2*DEV_ITEMS:
                break
        if len(complete) != 2*DEV_ITEMS:
            raise ValueError(f"Insufficient complete development data for {role}")
        dev[role] = complete

    def verified(offset, roles=ROLES):
        result = {}
        for role in roles:
            items = dev[role][offset:offset+DEV_ITEMS]
            generated = generate_greedy(backend, items, 512, [0]*len(items))
            result[role] = np.asarray(verify_completions(verifier, items, generated), dtype=bool)
        return result

    def losses():
        return {role: np.asarray([backend.measure_loss([item]) for item in dev[role][:DEV_ITEMS]])
                for role in ROLES}

    starts, predictions, before = {}, {}, {}
    for name in snapshots:
        backend.restore(snapshots[name])
        predictions[name] = verified(0, ("target_dev",))["target_dev"]
        starts[name] = {role: float(x.mean()) for role, x in verified(DEV_ITEMS).items()}
        before[name] = losses()
    fixed = ~predictions["m0"] & predictions["m1"]
    records = []
    for index, dose in enumerate(DOSES):
        optimizer_seed = seed + 20_000 + index
        effects, changes, timing = {}, {}, {}
        for name in snapshots:
            after, probe_info = probe(
                backend, snapshots[name], hashes[name], batches[dose], optimizer_seed, losses,
            )
            effects[name] = {r: before[name][r] - after[r] for r in ROLES}
            eval_start = time.perf_counter()
            result, full_info = probe(
                backend, snapshots[name], hashes[name], batches[dose], optimizer_seed,
                lambda: verified(DEV_ITEMS), steps=25,
            )
            changes[name] = {role: float(x.mean()-starts[name][role]) for role, x in result.items()}
            timing[name] = {"probe": probe_info, "full_update": full_info,
                            "full_update_and_scoring_seconds": time.perf_counter()-eval_start}
        gain = float(np.mean([x.mean() for x in effects["m1"].values()]))
        metric, penalty = score(gain, effects["m0"]["target_dev"], fixed, SMOKE_LAMBDA)
        records.append({"dose": dose, "batch": row["batches"][dose], "role_increments": changes,
                        "probe_gain": gain, "probe_penalty": penalty, "score": metric, "timing": timing})
    # Two 25-step updates each present eight examples per step. With 16 unique
    # examples, 50 steps from M0 matches the overall presentation budget.
    backend.config = replace(cfg, training=replace(cfg.training, max_steps=50))
    joint, train = probe(backend, snapshots["m0"], hashes["m0"], pooled,
                         seed+30_000, lambda: verified(DEV_ITEMS), steps=50)
    backend.config = cfg
    return {"passed": True, "job_id": os.environ["SLURM_JOB_ID"], "seed": seed,
            "development_only": True, "final_eval_read": False, "fixed_items": int(fixed.sum()),
            "dev_items_per_role_per_split": DEV_ITEMS, "parent_receipt_sha256": args.parent_sha256,
            "feasibility_sha256": file_digest(args.feasibility),
            "starts": starts, "draw": row["draw"], "doses": records,
            "pooled": {"training": train, "scores": {r: float(v.mean()) for r, v in joint.items()}},
            "elapsed_seconds": time.perf_counter()-stamp}


def main():
    require_compute_allocation()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "registry", "parent", "feasibility", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--home-cap", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--parent-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.validate_only:
        _, seed, row, batches, _ = validate_design(args)
        result = {"passed": True, "job_id": os.environ["SLURM_JOB_ID"], "seed": seed,
                  "draw": row["draw"], "dose_count": len(batches),
                  "feasibility_sha256": file_digest(args.feasibility)}
    else:
        if args.smoke is None:
            parser.error("GPU pilot requires --smoke")
        result = run(args)
    write_json(args.output, result)
    if args.validate_only:
        print(f"Validated three overlap doses and pooled control for {result['seed']}", flush=True)
    else:
        print(f"Timed paired probes, six updates and pooled control in {result['elapsed_seconds']:.1f} s",
              flush=True)


if __name__ == "__main__":
    main()
