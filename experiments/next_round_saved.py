"""Read-only next-round sensitivities; run only on Slurm CPU compute nodes."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from third_eye.cluster import require_compute_allocation
from third_eye.evaluation.final import load_final_manifest
from third_eye.evaluation.verifiers import extract_numeric
from third_eye.experiments.confirmation import checkpoints, read_scored_items
from forecasting_pilots import inspect_task
from third_eye.experiments.regime_revision import (
    effective_distance, factor_pairs, lenient_numeric, paired_bootstrap,
)
from third_eye.experiments.smollm_confirmation import matrix, NAMES, ROLES
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text())


def factors(directory, expected_hash):
    """Read one verified adapter in memory; never unpack archives onto home."""
    from safetensors.torch import load
    folder = Path(directory)
    state = read(folder / "state.json")
    if state["adapter_hash"] != expected_hash:
        raise ValueError("Norm checkpoint differs from scored adapter")
    config = state["config"]["training"]
    weights, raw_weights = {}, {}
    for name, expected in state["weights"].items():
        path = folder / name
        if path.exists():
            if file_digest(path) != expected:
                raise ValueError("Changed retained adapter")
            data = path.read_bytes()
        else:
            compressed = path.with_name(path.name + ".gz")
            if not compressed.exists():
                raise FileNotFoundError("Adapter and lossless archive absent: " + str(path))
            with gzip.open(compressed, "rb") as handle:
                data = handle.read(256 * 1024**2 + 1)
            if len(data) > 256 * 1024**2:
                raise ValueError("Adapter exceeds bounded in-memory archive limit")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Uncompressed adapter checksum differs")
        raw_weights.update(load(data))
    identity = hashlib.sha256()
    import torch
    for name, tensor in sorted(raw_weights.items()):
        identity.update(name.encode())
        identity.update(str((tensor.shape, tensor.dtype)).encode())
        identity.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        weights[name] = tensor.float().numpy()
    if identity.hexdigest() != expected_hash:
        raise ValueError("Retained effective-norm adapter hash differs")
    return factor_pairs(weights), config["alpha"] / config["rank"]


def load_states(plan_path, norm_records):
    plan = read(plan_path)
    output, hashes = [], {str(plan_path): file_digest(plan_path)}
    for task in plan["tasks"]:
        folder = Path(task["output"])
        done = read(folder / "completed.json")
        decision = read(folder / "commitment.json")
        if not done["passed"] or done["plan_sha256"] != file_digest(plan_path):
            raise ValueError("Complete immutable parent required")
        if file_digest(folder / "commitment.json") != done["commitment_sha256"]:
            raise ValueError("Frozen development commitment changed")
        _, final = load_final_manifest(task["final_manifest"], task["manifest"])
        rows, _ = inspect_task({**task, "output": str(folder / "collection")},
                              [50_000_000, 100_000_000], plan["source_tree_sha256"])
        paths = checkpoints(rows)
        strict = {"stream": task["stream"], "seed": task["seed"],
                  "actions": decision["actions"], "artifacts": {},
                  "ids": {r: [(e.id, e.prompt_hash) for e in xs] for r, xs in final.items()}}
        lenient = deepcopy(strict)
        for name, checkpoint, adapter_hash in paths:
            dest = folder / "scoring" / name
            for filename, key in [("items.jsonl", "items_sha256"),
                                  ("scores.json", "scores_sha256")]:
                actual = file_digest(dest / filename)
                if actual != done["scores"][name][key]:
                    raise ValueError("Immutable scored artifact changed")
                hashes[str(dest / filename)] = actual
            a = read_scored_items(dest, final, adapter_hash, done["commitment_sha256"])
            strict["artifacts"][name] = a
            revised = {r: v.copy() for r, v in a.items()}
            if task["stream"].endswith("/math"):
                saved = [json.loads(line) for line in (dest/"items.jsonl").read_text().splitlines()]
                observed = [x for x in saved if x["role"] == "target_test"]
                for i, (ex, item) in enumerate(zip(final["target_test"], observed)):
                    prediction = lenient_numeric(item["completion"])
                    reference = extract_numeric(ex.answer)
                    revised["target_test"][i] = float(
                        prediction is not None and reference is not None and prediction == reference)
                    if a["target_test"][i] and not revised["target_test"][i]:
                        raise ValueError("Lenient sensitivity must preserve strict passes")
            lenient["artifacts"][name] = revised
        output.append((strict, lenient))
        # Explicitly distinguish first, incremental second, and cumulative norms.
        parent_dir, parent_hash = next((p, h) for n, p, h in paths if n == "parent")
        parent, parent_scale = factors(parent_dir, parent_hash)
        for k in range(3):
            first_dir, first_hash = next((p, h) for n, p, h in paths if n == f"m1_{k}")
            first, first_scale = factors(first_dir, first_hash)
            first_norm = effective_distance(parent, first, parent_scale, first_scale)
            for d in range(3):
                second_dir, second_hash = next((p, h) for n, p, h in paths if n == f"m2_{k}_{d}")
                second, second_scale = factors(second_dir, second_hash)
                incremental = effective_distance(first, second, first_scale, second_scale)
                cumulative = effective_distance(parent, second, parent_scale, second_scale)
                norm_records.append({
                    "stream": task["stream"], "seed": task["seed"], "candidate": k,
                    "draw": d, "candidate_weight": decision["actions"]["forced"][k],
                    "first_update": first_norm["frobenius_norm"],
                    "second_increment": incremental["frobenius_norm"],
                    "cumulative": cumulative["frobenius_norm"],
                    "second_by_layer": incremental["layers"],
                })
            del first
        del parent
        print(json.dumps({"complete_parent": task["stream"], "seed": task["seed"]}), flush=True)
    return output, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_compute_allocation()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    pairs, hashes, norms = [], {}, []
    for path in args.plan:
        p, h = load_states(Path(path), norms)
        pairs.extend(p)
        hashes.update(h)
    states, values = [], []
    names = ["strict_"+x for x in NAMES] + ["lenient_"+x for x in NAMES]
    for strict, lenient in pairs:
        states.append(strict)
        values.append({r: np.concatenate([matrix(strict, r), matrix(lenient, r)])
                       for r in ROLES})
    report = {
        "scope": "Exploratory saved-outcome sensitivities; original confirmatory reports unchanged",
        **source_inventory(), "input_hashes": hashes, "parents": len(states),
        "intervals": paired_bootstrap(states, values, names),
        "lenient_scope": "GSM8K numeric final-line extraction only; other benchmark scores unchanged",
        "norm_interpretation": "Effective weight Frobenius norms; no causal mediation claim",
        "norms": norms,
        "bootstrap": "5000 stratified run/shared-item draws; three continuations averaged per candidate",
    }
    write_json(dest / "report.json", report)
    write_json(dest / "completed.json", {"passed": True,
        "report_sha256": file_digest(dest/"report.json"), "parents": len(states)})


if __name__ == "__main__":
    main()
