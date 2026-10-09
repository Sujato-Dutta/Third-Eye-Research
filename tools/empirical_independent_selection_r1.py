"""Independent-seed and disjoint-item selection evaluation from frozen E2 controls."""

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_noise_review_v1 import ROLES, matrix, inspect

FRESH = ("fresh_m2_50000000", "fresh_m2_100000000")
FUTURES = ("original_m2", *FRESH)
NAMESPACE = "third-eye-independent-selection-r1"


def weights(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Three finite candidate utilities required")
    selected = np.round(values, 12) == np.round(values, 12).max()
    return selected / selected.sum()


def compare(immediate, selector_future, heldout_future):
    greedy, future = weights(immediate), weights(selector_future)
    heldout_future = np.asarray(heldout_future, dtype=float)
    weights(heldout_future)  # Validate shape/finite values without selecting on them.
    a, b, random = float(greedy @ heldout_future), float(future @ heldout_future), float(heldout_future.mean())
    return dict(
        independent_future_minus_immediate=b - a,
        independent_future_utility=b,
        immediate_future_utility=a,
        random_future_utility=random,
        immediate_minus_random=a - random,
        future_minus_random=b - random,
        immediate_minus_hold=a,
    )


def item_folds(record):
    result = {}
    for role in ROLES:
        items = record["parent"]["items"][role]
        identities = [(str(x["id"]), x["prompt_sha256"]) for x in items]
        if len(items) < 2 or len({x[0] for x in identities}) != len(items):
            raise ValueError("Unique nonempty role items required")
        # The assignment is independent of predictions and identical wherever
        # role/item identities recur; states do not obtain new item samples.
        order = sorted(range(len(items)), key=lambda i: hashlib.sha256(
            json.dumps([NAMESPACE, role, *identities[i]], separators=(",", ":")).encode()
        ).hexdigest())
        middle = len(order) // 2
        result[role] = (np.asarray(order[:middle]), np.asarray(order[middle:]))
    return result


def utilities(arrays, parent, folds=None, fold=0):
    parts = []
    for role in ROLES:
        indices = slice(None) if folds is None else folds[role][fold]
        parts.append(arrays[role][:, indices].mean(1) - parent[role][indices].mean())
    return np.mean(parts, axis=0)


def mean_dict(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}


def analyse(record):
    names = ("original_m1", *FUTURES)
    if len(record["candidates"]) != 3 or len({c["candidate_id"] for c in record["candidates"]}) != 3:
        raise ValueError("Complete unique K=3 candidates required")
    for c in record["candidates"]:
        seeds = [c["conditions"][name]["seed"] for name in FRESH]
        if seeds[0] == seeds[1] or seeds[1] - seeds[0] != 50000000:
            raise ValueError("Fixed fresh continuation offsets required")
    folds = item_folds(record)
    parent = {role: np.asarray([x["correct"] for x in record["parent"]["items"][role]], float) for role in ROLES}
    if any(not np.isin(v, [0, 1]).all() for v in parent.values()):
        raise ValueError("Invalid parent correctness")
    arrays = {name: {role: matrix(record, name, role) for role in ROLES} for name in names}
    full = {name: utilities(v, parent) for name, v in arrays.items()}
    split = {name: [utilities(v, parent, folds, f) for f in (0, 1)] for name, v in arrays.items()}
    primary_folds = [compare(split["original_m1"][f], split[FRESH[0]][f], split[FRESH[1]][1-f]) for f in (0, 1)]
    primary = mean_dict(primary_folds)
    ordered = {}
    for selector, evaluator in itertools.permutations(FUTURES, 2):
        ordered[f"{selector}->{evaluator}"] = mean_dict([
            compare(split["original_m1"][f], split[selector][f], split[evaluator][1-f]) for f in (0, 1)
        ])
    one, two = [], []
    for heldout in FUTURES:
        training = [name for name in FUTURES if name != heldout]
        for f in (0, 1):
            test = split[heldout][1-f]
            one.append(float(np.mean([weights(split[name][f]) @ test for name in training])))
            two.append(float(weights(np.mean([split[name][f] for name in training], axis=0)) @ test))
    return dict(
        state_id=record["state_id"], trajectory_id=record["trajectory_id"], stream=record["stream"],
        primary=primary, primary_folds=primary_folds,
        independent_seed_shared_items=compare(full["original_m1"], full[FRESH[0]], full[FRESH[1]]),
        ordered_crossfit_pairs=ordered,
        secondary_one_vs_two=dict(
            one_continuation_heldout_utility=float(np.mean(one)),
            two_continuations_heldout_utility=float(np.mean(two)),
            two_minus_one=float(np.mean(np.asarray(two) - one)),
        ),
        fold_identities={role: [[record["parent"]["items"][role][i]["id"] for i in group] for group in folds[role]] for role in ROLES},
        fresh_terminal_fraction=float(np.mean([1-c["conditions"][name]["continuation_available"] for c in record["candidates"] for name in FRESH])),
    )


def intervals(rows, values):
    if len(rows) != 8 or len({r["trajectory_id"] for r in rows}) != 8:
        raise ValueError("Eight independent parent trajectories required")
    streams = sorted({r["stream"] for r in rows})
    groups = [np.asarray([i for i, r in enumerate(rows) if r["stream"] == stream]) for stream in streams]
    if len(groups) != 4 or any(len(g) != 2 for g in groups):
        raise ValueError("Balanced two-parent/four-stream cohort required")
    rng = np.random.default_rng(20261009)
    draws = np.column_stack([rng.choice(group, (5000, 2), replace=True) for group in groups])
    result = {}
    for name in values[0]:
        data = np.asarray([v[name] for v in values], float)
        result[name] = dict(mean=float(data.mean()), ci95=np.quantile(data[draws].mean(1), [0.025, 0.975]).tolist(), independent_trajectories=8)
    return result


def main():
    from empirical_checkpoint_v3 import OUT, frozen, load_noise_input
    from third_eye.io import file_digest, write_json
    frozen()  # Includes the Slurm compute-node guard and frozen core methods.
    output = OUT / "independent_selection_r1"
    release_path = output / "release.json"
    release = json.loads(release_path.read_text())
    if release["core_release_sha256"] != file_digest(OUT / "release.json") or any(file_digest(ROOT / p) != h for p, h in release["files"].items()):
        raise RuntimeError("R1 source or prerequisite changed")
    if (output / "report.json").exists():
        raise RuntimeError("Preserve prior analysis; never overwrite")
    records, missing, input_hashes = [], [], {}
    for index in range(8):
        p = OUT / "noise" / f"task_{index}/completed.json"
        if not p.exists():
            missing.append(index)
            continue
        record = json.loads(p.read_text())
        if record["status"] != "complete" or record["noise_release_sha256"] != release["core_release_sha256"]:
            raise RuntimeError("Invalid noise source identity")
        task = load_noise_input(index)
        if (record["state_id"], record["trajectory_id"], record["stream"]) != (task["state_id"], task["trajectory_id"], task["stream"]):
            raise RuntimeError("Noise parent identity differs")
        for role, expected in zip(ROLES, (64, 64, 256)):
            if len(record["parent"]["items"][role]) != expected:
                raise RuntimeError("Evaluation budget differs")
        input_hashes[str(p)] = file_digest(p)
        records.append(record)
    report = dict(status="incomplete_review_stop" if missing else "science_review_required", missing_indices=missing, complete_parents=len(records), input_sha256=input_hashes, release_sha256=file_digest(release_path), new_gpu_experiments=0)
    if not missing:
        # Native inspect audits terminal preservation, fixed optimizer budgets,
        # composition and aligned items before any new comparison is published.
        matched = [inspect(record) for record in records]
        results = [analyse(record) for record in records]
        noise_keys = ("original_h1_h2_reversal", "fixed_candidate_optimizer_reversal", "fixed_continuation_optimizer_reversal", "fresh_continuation_reversal", "reversal_minus_fixed_continuation_noise", "reversal_minus_fresh_continuation_variation")
        report.update(
            primary_independent_seed_disjoint_items=intervals(results, [r["primary"] for r in results]),
            secondary_independent_seed_shared_items=intervals(results, [r["independent_seed_shared_items"] for r in results]),
            secondary_one_vs_two=intervals(results, [r["secondary_one_vs_two"] for r in results]),
            matched_noise_comparisons=intervals(results, [{k:r[k] for k in noise_keys} for r in matched]),
            parent_results=results,
        )
    report["interpretation"] = "Primary selector uses fresh +50M seed, scorer uses fresh +100M seed on disjoint item halves, averaging both fold orientations. Signed gap may be negative: this is not regret against a true oracle. Eight trajectories and shared finite evaluation pools limit inference. Secondary one-vs-two continuation comparison cannot determine a universal replicate requirement. No causal or acceptance claim."
    write_json(output / "report.json", report)
    import tarfile
    target = Path("/work/11617/sujato_ts/vista/third_eye_results/independent_selection_r1.tar.gz")
    if target.exists():
        raise RuntimeError("Preserve prior durable archive")
    partial = target.with_suffix(".partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
        archive.add(output, arcname=str(output.relative_to(ROOT)))
        for p in release["files"]:
            archive.add(ROOT / p, arcname=p)
    partial.replace(target)
    write_json(output / "archive_receipt.json", dict(path=str(target), sha256=file_digest(target), bytes=target.stat().st_size))


if __name__ == "__main__":
    main()
