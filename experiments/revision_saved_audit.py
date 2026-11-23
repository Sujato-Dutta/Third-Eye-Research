"""Audit sealed positive-regime scores without transferring model weights."""

import argparse
import json
from pathlib import Path

import numpy as np

from third_eye.cluster import require_compute_allocation
from third_eye.evaluation.final import load_final_manifest
from third_eye.evaluation.verifiers import extract_numeric
from third_eye.experiments.positive_regime import DECLARATION, ROLES
from third_eye.experiments.regime_revision import lenient_numeric
from third_eye.io import digest, file_digest, write_json

DRAW_COUNT = DECLARATION["continuation_draws"]
STATES = ["m0", "m1"] + [
    f"d{draw}_{arm}" for draw in range(DRAW_COUNT) for arm in ("native", "direct")
]
ACTIONS = ("carry", "skip", "stop")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score_parent(folder, seed, expected, receipt, final_manifest_sha256):
    completed_path = folder / "completed.json"
    if file_digest(completed_path) != receipt:
        raise ValueError(f"Parent {seed} does not match the integrated review")
    done = read(completed_path)
    seal_path = folder / "commitment.json"
    seal = read(seal_path)
    if (not done["passed"] or done["seed"] != seed or
            seal["final_manifest_sha256"] != final_manifest_sha256 or
            file_digest(seal_path) != done["commitment_sha256"]):
        raise ValueError(f"Parent {seed} has an invalid completion seal")
    ordered = [(role, item) for role in ROLES for item in expected[role]]
    output = {role: [] for role in ROLES}
    formats = {}
    for name in STATES:
        directory = folder / "scoring" / name
        hashes = done["scores"][name]
        score_path, item_path = directory / "scores.json", directory / "items.jsonl"
        if (file_digest(score_path) != hashes["scores_sha256"] or
                file_digest(item_path) != hashes["items_sha256"]):
            raise ValueError(f"Altered scored checkpoint {seed}/{name}")
        meta = read(score_path)
        if (meta["adapter_hash"] != seal["states"][name]["adapter_hash"] or
                meta["commitment_sha256"] != done["commitment_sha256"]):
            raise ValueError(f"Wrong score commitment for {seed}/{name}")
        outcome = {role: [] for role in ROLES}
        with item_path.open(encoding="utf-8") as lines:
            rows = (json.loads(line) for line in lines if line.strip())
            strict_failures = recovered = recovered_correct = target_words = 0
            for row, (role, item) in zip(rows, ordered, strict=True):
                if (row["role"], row["id"]) != (role, item.id) or type(row["passed"]) is not bool:
                    raise ValueError(f"Mismatched evaluation item in {seed}/{name}")
                outcome[role].append(row["passed"])
                if name in ("m0", "m1") and role == "target_test":
                    completion = row["completion"]
                    target_words += len(completion.split())
                    if extract_numeric(completion) is None:
                        strict_failures += 1
                        alternative = lenient_numeric(completion)
                        recovered += alternative is not None
                        recovered_correct += alternative is not None and alternative == extract_numeric(item.answer)
        for role in ROLES:
            values = np.asarray(outcome[role], dtype=bool)
            if abs(values.mean() - meta["scores"][role]) > 1e-12:
                raise ValueError("Saved scores and item outcomes disagree")
            output[role].append(values)
        if name in ("m0", "m1"):
            formats[name] = {
                "items": len(expected["target_test"]),
                "strict_extraction_failures": strict_failures,
                "lenient_extractions_recovered": recovered,
                "lenient_correct_answers_recovered": recovered_correct,
                "total_words": target_words,
            }
    return output, formats


def collect(args):
    request = read(args.request)
    integrated = read(args.integrated)
    if not integrated["passed"] or sorted(map(int, integrated["parent_completed_sha256"])) != DECLARATION["seeds"]:
        raise ValueError("Complete integrated confirmation receipt required")
    input_hashes = {field: request[field + "_sha256"] for field in ("manifest", "final_manifest")}
    for field, sha in input_hashes.items():
        if file_digest(request[field]) != sha:
            raise ValueError(f"Frozen {field} differs")
    _, expected = load_final_manifest(request["final_manifest"], request["manifest"])
    if {r: len(expected[r]) for r in ROLES} != DECLARATION["evaluation_items"]:
        raise ValueError("Unexpected evaluation suite size")
    records = {role: [] for role in ROLES}
    formats = {}
    for seed in args.seeds:
        output, fmt = score_parent(
            args.parents / str(seed), seed, expected,
            integrated["parent_completed_sha256"][str(seed)],
            input_hashes["final_manifest"],
        )
        for role in ROLES:
            records[role].append(output[role])
        formats[str(seed)] = fmt
        print(f"Verified saved scores for {seed}", flush=True)
    metadata = {
        "item_hashes": {r: digest([x.id for x in expected[r]]) for r in ROLES},
        "input_hashes": input_hashes,
        "development_final_disjoint": True,
        "seeds": args.seeds,
        "format": formats,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.with_suffix(".partial").open("xb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), **records)
    args.output.with_suffix(".partial").replace(args.output)


def paired_data(values):
    """Expose the same draw ordering for carry, skip, and stop."""
    if values.ndim != 3 or values.shape[1] != len(STATES) or values.dtype != np.bool_:
        raise ValueError("Expected seed × checkpoint × item binary outcomes")
    m0, m1 = values[:, 0], values[:, 1]
    native, direct = values[:, 2::2], values[:, 3::2]
    repeated = np.broadcast_to(m1[:, None], native.shape)
    return m0, m1, np.stack((native, direct, repeated), axis=2)


def choice_scores(by_role):
    """Rotate choice draws and disjoint item halves, then score held-out cells."""
    n = next(iter(by_role.values())).shape[0]
    gains = {role: np.zeros((n, 4, values.shape[-1])) for role, values in by_role.items()}
    outcomes = {role: paired_data(values)[2].astype(float) for role, values in by_role.items()}
    for choice_draw in range(DRAW_COUNT):
        test_draws = [d for d in range(DRAW_COUNT) if d != choice_draw]
        for half in range(2):
            choice = np.mean([x[:, choice_draw, :, half::2].mean(axis=-1)
                              for x in outcomes.values()], axis=0)
            adaptive = choice.argmax(axis=1)
            # Select the fixed comparator on other seeds and the choice items/draw.
            fixed = (choice.sum(axis=0) - choice).argmax(axis=1)
            for role, x in outcomes.items():
                test = x[:, test_draws, :, 1-half::2].mean(axis=1)
                chosen = test[np.arange(n), adaptive]
                gains[role][:, 0, 1-half::2] += (chosen - test[np.arange(n), fixed]) / DRAW_COUNT
                gains[role][:, 1:, 1-half::2] += (chosen[:, None] - test) / DRAW_COUNT
    return gains


def closed_intervals(effects, reps=5000):
    """Paired bootstrap over seeds and shared items, with joint role averaging."""
    rng = np.random.default_rng(20261122)
    n = next(iter(effects.values())).shape[0]
    seed_weights = rng.multinomial(n, np.full(n, 1/n), size=reps) / n
    points, boots = [], []
    for values in effects.values():
        items = values.shape[-1]
        points.append(values.mean(axis=(0, 2)))
        boot = np.empty((reps, values.shape[1]))
        for start in range(0, reps, 64):
            stop = min(start+64, reps)
            item_weights = rng.multinomial(items, np.full(items, 1/items), size=stop-start) / items
            scores = np.einsum("bn,nki->bki", seed_weights[start:stop], values, optimize=True)
            boot[start:stop] = np.einsum("bki,bi->bk", scores, item_weights, optimize=True)
        boots.append(boot)
    return [{"mean_pp": float(p*100), "ci95_pp": [float(v*100) for v in ci]}
            for p, ci in zip(np.mean(points, axis=0), np.quantile(np.mean(boots, axis=0), [.025, .975], axis=0).T)]


def summarize(by_role):
    flips, levels = {}, []
    for role, data in by_role.items():
        m0, m1, arms = paired_data(data)
        native, direct = arms[:, :, 0], arms[:, :, 1]
        direct_fixes = direct & ~m0[:, None]
        already_fixed = direct_fixes & m1[:, None]
        unfixed = direct_fixes & ~native
        denominator = int(direct_fixes.sum())
        flips[role] = {
            "direct_fixes": denominator,
            "already_fixed_by_m1": int(already_fixed.sum()),
            "already_fixed_share": float(already_fixed.sum()/denominator) if denominator else None,
            "still_unfixed_after_native": int(unfixed.sum()),
            "still_unfixed_share": float(unfixed.sum()/denominator) if denominator else None,
            "native_new_fixes": int((native & ~m1[:, None]).sum()),
            "native_new_breaks": int((~native & m1[:, None]).sum()),
            "direct_new_breaks": int((~direct & m0[:, None]).sum()),
        }
        levels.append(arms.mean(axis=(1, 3)) - m0.mean(axis=1)[:, None])
    levels = np.mean(levels, axis=0)
    estimates = closed_intervals(choice_scores(by_role))
    return {
        "parents": int(levels.shape[0]),
        "item_flips": flips,
        "fixed_action_gains_pp": dict(zip(ACTIONS, (levels.mean(axis=0)*100).tolist())),
        "optimistic_same_outcome_oracle_gap_pp": float(
            100*(levels.max(axis=1).mean()-levels.mean(axis=0).max())),
        "independent_choice": dict(zip(("vs_out_of_fold_fixed",) +
                                      tuple("vs_"+action for action in ACTIONS), estimates)),
        "choice_scope": "Held-out items and draws; per-seed choice uses continuation outcomes, not a deployable probe",
    }


def review(args):
    by_seed, formats = {}, {}
    signature = None
    for path in args.inputs:
        with np.load(path, allow_pickle=False) as bundle:
            metadata = json.loads(str(bundle["metadata"]))
            current = (metadata["item_hashes"], metadata["input_hashes"])
            if signature not in (None, current):
                raise ValueError("Cross-cluster manifests or item identities differ")
            signature = current
            for index, seed in enumerate(metadata["seeds"]):
                if seed in by_seed:
                    raise ValueError(f"Seed exported twice: {seed}")
                by_seed[seed] = {role: bundle[role][index] for role in ROLES}
            formats.update(metadata["format"])
    if sorted(by_seed) != DECLARATION["seeds"]:
        raise ValueError("All 20 prescribed seeds are required")
    report = summarize({role: np.stack([by_seed[seed][role] for seed in sorted(by_seed)])
                        for role in ROLES})
    report.update({"passed": True, "format_by_seed": formats, "development_final_disjoint": True,
                   "input_hashes": signature[1], "input_artifacts": [file_digest(path) for path in args.inputs]})
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "format_by_seed"}), flush=True)


def main():
    require_compute_allocation()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("collect")
    for name in ("request", "integrated", "parents", "output"):
        exporter.add_argument(f"--{name}", type=Path, required=True)
    exporter.add_argument("--seeds", type=int, nargs="+", choices=DECLARATION["seeds"], required=True)
    reviewer = commands.add_parser("review")
    reviewer.add_argument("--inputs", type=Path, nargs="+", required=True)
    reviewer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        review(args)


if __name__ == "__main__":
    main()
