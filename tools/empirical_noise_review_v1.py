# ruff: noqa: E402
"""Paired item uncertainty and matched-seed reliability; no gate replacement."""

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_noise_v1 import BASE, HEADS, ROLES, frozen
from third_eye.io import file_digest, write_json


def winners(values):
    values = np.round(np.asarray(values), 12)
    return set(np.flatnonzero(values == values.max()))


def reversal(one, two):
    return float(not winners(one).intersection(winners(two)))


def regret(one, two):
    return float(np.max(two) - np.mean(two[list(winners(one))]))


def matrix(record, condition, role):
    parent = record["parent"]["items"][role]
    identity = [(x["id"], x["prompt_sha256"]) for x in parent]
    arrays = []
    for c in record["candidates"]:
        items = c["conditions"][condition]["items"][role]
        if [(x["id"], x["prompt_sha256"]) for x in items] != identity:
            raise RuntimeError(
                "Candidate outcomes are not aligned with original role items"
            )
        if any(x["correct"] not in (0, 1) for x in items):
            raise RuntimeError("Invalid verifier outcome")
        arrays.append([x["correct"] for x in items])
    if len(arrays) != 3:
        raise RuntimeError("Reliability requires complete K=3")
    return np.array(arrays, dtype=float)


def inspect(record):
    conditions = list(record["candidates"][0]["conditions"])
    if any(set(c["conditions"]) != set(conditions) for c in record["candidates"]):
        raise RuntimeError("Incomplete matched condition coverage")
    if len({c["batch_hash"] for c in record["candidates"]}) != 3:
        raise RuntimeError("Candidate compositions identical")
    arrays = {
        name: {role: matrix(record, name, role) for role in ROLES}
        for name in conditions
    }
    parent = np.array([record["parent"]["metrics"][h] for h in HEADS])
    heads = {
        name: np.column_stack([arrays[name][r].mean(1) for r in ROLES]) - parent
        for name in conditions
    }
    utilities = {name: values.mean(1) for name, values in heads.items()}
    one, two = utilities["original_m1"], utilities["original_m2"]
    fresh = [n for n in conditions if n.startswith("fresh_m2_")]
    if len(fresh) != 2:
        raise RuntimeError("Need exactly two new matched continuation seeds")
    average_future = np.mean([two, *[utilities[n] for n in fresh]], axis=0)
    greedy = list(winners(one))
    result = dict(
        state_id=record["state_id"],
        trajectory_id=record["trajectory_id"],
        stream=record["stream"],
        generation=record["generation"],
        original_h1_h2_reversal=reversal(one, two),
        original_greedy_regret=regret(one, two),
        fixed_candidate_optimizer_reversal=reversal(one, utilities["optimizer_m1"]),
        fixed_continuation_optimizer_reversal=reversal(two, utilities["optimizer_m2"]),
        h1_vs_three_seed_average_h2_reversal=reversal(one, average_future),
        greedy_three_seed_future_utility=float(average_future[greedy].mean()),
        random_three_seed_future_utility=float(average_future.mean()),
        oracle_three_seed_future_utility=float(average_future.max()),
        hold_parameters_future_utility=0.0,
        greedy_minus_random=float(
            average_future[greedy].mean() - average_future.mean()
        ),
        original_future_utility=two.tolist(),
        three_seed_future_utility=average_future.tolist(),
    )
    result["fresh_continuation_reversal"] = float(
        np.mean([reversal(two, utilities[n]) for n in fresh])
    )
    result["reversal_minus_fixed_continuation_noise"] = (
        result["original_h1_h2_reversal"]
        - result["fixed_continuation_optimizer_reversal"]
    )
    result["reversal_minus_fresh_continuation_variation"] = (
        result["original_h1_h2_reversal"] - result["fresh_continuation_reversal"]
    )
    terminal = []
    for c in record["candidates"]:
        for name in fresh + ["optimizer_m2"]:
            value = c["conditions"][name]
            if not value["continuation_available"]:
                original = c["conditions"]["original_m1"]
                if (
                    value["adapter_hash"] != original["adapter_hash"]
                    or value["items"] != original["items"]
                    or value["training"]["optimizer_steps"] != 0
                ):
                    raise RuntimeError("Terminal branch does not preserve M2=M1")
            elif value["training"]["optimizer_steps"] != 50:
                raise RuntimeError("Nonterminal optimizer budget differs")
            terminal.append(1 - value["continuation_available"])
    result["fresh_terminal_fraction"] = float(
        np.mean(
            [
                1 - c["conditions"][n]["continuation_available"]
                for c in record["candidates"]
                for n in fresh
            ]
        )
    )
    # Shared item draws across all candidates and horizons preserve paired covariance.
    rng = np.random.default_rng(42)
    draws = {
        role: rng.integers(
            0,
            arrays["original_m1"][role].shape[1],
            (2000, arrays["original_m1"][role].shape[1]),
        )
        for role in ROLES
    }
    boot = {
        name: np.mean([arrays[name][r][:, draws[r]].mean(2).T for r in ROLES], axis=0)
        for name in ("original_m1", "original_m2")
    }
    reversals = [
        reversal(a, b) for a, b in zip(boot["original_m1"], boot["original_m2"])
    ]
    regrets = [regret(a, b) for a, b in zip(boot["original_m1"], boot["original_m2"])]
    result["paired_item_bootstrap"] = dict(
        draws=2000,
        seed=42,
        reversal_fraction=float(np.mean(reversals)),
        regret_ci95=np.quantile(regrets, [0.025, 0.975]).tolist(),
        h1_original_winner_stability=float(
            np.mean([bool(winners(one) & winners(a)) for a in boot["original_m1"]])
        ),
        h2_original_winner_stability=float(
            np.mean([bool(winners(two) & winners(b)) for b in boot["original_m2"]])
        ),
        interpretation="Finite-item uncertainty conditional on fixed predictions; not a training-noise null or probability of a population claim",
    )
    return result


def aggregate(rows):
    if len({r["trajectory_id"] for r in rows}) != len(rows):
        raise RuntimeError(
            "Noise parent states must belong to independent trajectories"
        )
    keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(rows), (5000, len(rows)))
    return {
        k: dict(
            mean=float(np.mean([r[k] for r in rows])),
            ci95=np.quantile(
                np.array([r[k] for r in rows])[ix].mean(1), [0.025, 0.975]
            ).tolist(),
            independent_trajectories=len(rows),
        )
        for k in keys
    }


def main():
    frozen()
    records, missing = [], []
    for index in range(12):
        path = BASE / "noise" / f"task_{index}/completed.json"
        if not path.exists():
            missing.append(index)
            continue
        r = json.loads(path.read_text())
        if r["status"] != "complete" or r["noise_release_sha256"] != file_digest(
            BASE / "noise_release.json"
        ):
            raise RuntimeError("Unexpected noise result release")
        records.append(inspect(r))
    if missing:
        write_json(
            BASE / "noise_incomplete.json",
            dict(
                status="incomplete_review_stop",
                missing=missing,
                complete_states=len(records),
            ),
        )
        raise RuntimeError(
            f"Noise study incomplete: {missing}; retain failures, do not replace states"
        )
    output = BASE / "noise_review"
    output.mkdir(exist_ok=False)
    write_json(
        output / "review.json",
        dict(
            status="noise_science_review_required",
            states=12,
            all=aggregate(records),
            streams={
                s: aggregate([r for r in records if r["stream"] == s])
                for s in sorted({r["stream"] for r in records})
            },
            parent_results=records,
            replication_released=False,
            gate1_or_gate2_replaced=False,
            interpretation="Matched repeatability and uncertainty evidence; no automatic scientific pass/fail threshold",
        ),
    )


if __name__ == "__main__":
    main()
