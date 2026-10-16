"""Paired item/seed selection comparisons for a declared LR intervention."""

import hashlib
import json

import numpy as np

from third_eye.statistics.inference import paired_comparison, holm


HEADS = ("target", "ood", "retention")
ROLES = ("target_dev", "ood_dev", "retention_dev")
OFFSETS = (50_000_000, 100_000_000, 150_000_000)


def intervention_export(record):
    """Remove inherited control consequences from the intervention artifact.

    Only measured M1 and the actual continuation_replicates are outcomes.
    The reference seed defines offsets; it is not an observed intervention M2.
    Removing precommit also prevents using baseline-LR probes as new features.
    """
    import copy

    result = copy.deepcopy(record)
    result["reference_continuation_seed"] = result.pop("continuation")["seed"]
    result.pop("continuation_available", None)
    result.pop("continuation_adapter_hash", None)
    result.pop("precommit", None)
    result["labels"] = {"h1": result["labels"]["h1"]}
    result["evaluation"].pop("t2", None)
    result["evaluation_artifacts"].pop("t2", None)
    result["runtime"].pop("continuation", None)
    result["artifact_kind"] = "paired_candidate_lr_intervention"
    return result


def choose(values):
    values = np.asarray(values, float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Finite complete K=3 utility required")
    tied = np.round(values, 12) == np.round(values, 12).max()
    return tied / tied.sum()


def folds(parent):
    result = {}
    for role in ROLES:
        items = parent["items"][role]
        ids = [(x["id"], x["prompt_sha256"]) for x in items]
        if len(items) < 2 or len(set(ids)) != len(items):
            raise ValueError("Unique evaluation items required")
        order = sorted(
            range(len(items)),
            key=lambda i: hashlib.sha256(
                json.dumps(
                    ["third-eye-lr-mechanism-20261013", role, *ids[i]],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        result[role] = (order[: len(order) // 2], order[len(order) // 2 :])
    return result


def matrix(artifacts, parent, assignment, fold):
    """Three-head deltas by candidate, with checkpoint-aligned item identities."""
    result = []
    for artifact in artifacts:
        values = []
        for role in ROLES:
            p = parent["items"][role]
            a = artifact["items"][role]
            if [(x["id"], x["prompt_sha256"]) for x in a] != [
                (x["id"], x["prompt_sha256"]) for x in p
            ]:
                raise ValueError("Evaluation items/order changed")
            if any(
                type(x["correct"]) is not int or x["correct"] not in (0, 1)
                for x in [*p, *a]
            ):
                raise ValueError("Invalid item verdict")
            indices = assignment[role][fold]
            values.append(
                float(np.mean([a[i]["correct"] - p[i]["correct"] for i in indices]))
            )
        result.append(values)
    return np.asarray(result)


def state_comparison(
    parent, baseline_m1, baseline_m2, intervention_m1=None, intervention_m2=None
):
    """Select on one item half, score on the other and a held-out draw.

    First two declared draws select lookahead; third scores. Both item-fold
    orientations are averaged inside a state, never counted as new units.
    """
    assignment = folds(parent)
    if len(baseline_m2) != 3 or any(len(draw) != 3 for draw in baseline_m2):
        raise ValueError("Three draws of complete K=3 candidates required")
    if intervention_m1 is not None and (
        intervention_m2 is None
        or len(intervention_m2) != 3
        or any(len(d) != 3 for d in intervention_m2)
    ):
        raise ValueError("Matched intervention draws required")
    result = []
    for fold in (0, 1):
        h1 = matrix(baseline_m1, parent, assignment, fold)
        lookahead = np.mean(
            [matrix(draw, parent, assignment, fold) for draw in baseline_m2[:2]], axis=0
        )
        future = matrix(baseline_m2[2], parent, assignment, 1 - fold)
        greedy, ahead = choose(h1.mean(1)), choose(lookahead.mean(1))
        # Fixed inexpensive rule: positive equal-head immediate utility AND no
        # retention decrease; ties fractionally weighted, otherwise hold.
        eligible = (h1.mean(1) > 0) & (h1[:, 2] >= 0)
        guarded = (
            choose(np.where(eligible, h1.mean(1), -1e6))
            if eligible.any()
            else np.zeros(3)
        )
        row = dict(
            future_minus_immediate=float((ahead - greedy) @ future.mean(1)),
            immediate_minus_hold=float(greedy @ future.mean(1)),
            future_minus_random=float(ahead @ future.mean(1) - future.mean()),
            guarded_minus_immediate=float((guarded - greedy) @ future.mean(1)),
            guarded_minus_hold=float(guarded @ future.mean(1)),
            guarded_retention_minus_immediate=float((guarded - greedy) @ future[:, 2]),
            guarded_update_probability=float(guarded.sum()),
        )
        if intervention_m1 is not None:
            small_h1 = matrix(intervention_m1, parent, assignment, fold)
            small_future = matrix(intervention_m2[2], parent, assignment, 1 - fold)
            small_greedy = choose(small_h1.mean(1))
            # Primary fixes baseline-greedy choice, isolating the effect of the
            # LR intervention on the same candidate, rather than reselecting.
            row.update(
                half_lr_same_candidate_minus_baseline=float(
                    greedy @ (small_future - future).mean(1)
                ),
                half_lr_same_candidate_retention_delta=float(
                    greedy @ (small_future - future)[:, 2]
                ),
                half_lr_reselected_minus_baseline=float(
                    small_greedy @ small_future.mean(1) - greedy @ future.mean(1)
                ),
                half_lr_mean_all_draws_delta=float(
                    np.mean(
                        [
                            matrix(intervention_m2[d], parent, assignment, 1 - fold)
                            - matrix(baseline_m2[d], parent, assignment, 1 - fold)
                            for d in range(3)
                        ]
                    )
                ),
            )
        result.append(row)
    return {k: float(np.mean([r[k] for r in result])) for k in result[0]}


def aggregate(states):
    if len(states) != 8 or len({s["trajectory_id"] for s in states}) != 8:
        raise ValueError("Eight complete distinct parent trajectories required")
    streams = {s["stream"] for s in states}
    if len(streams) != 4 or any(
        sum(s["stream"] == stream for s in states) != 2 for stream in streams
    ):
        raise ValueError("Balanced four-stream evidence required")
    keys = states[0]["comparison"].keys()
    results = {
        key: paired_comparison(
            [s["comparison"][key] for s in states], [0] * 8, seed=20261013
        )
        for key in keys
    }
    adjusted = holm(
        {
            k: v["paired_permutation_p"]
            for k, v in results.items()
            if "probability" not in k
        }
    )
    for key, value in adjusted.items():
        results[key]["holm_p"] = value
    return dict(
        comparisons=results,
        independent_trajectories=8,
        development_only=True,
        automatic_expansion=False,
        gate2_passed=None,
        limitations="Previously inspected developmental parents and shared finite item pools; folds, candidates and draws are not independent units. LR intervention changes candidate training only; endogenous continuation-data changes are part of its total effect. Eight parents cannot establish population-wide safety or forecasting superiority.",
    )
