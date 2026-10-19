"""Native three-draw development panels; no fabricated crossed outcomes."""

import numpy as np

from third_eye.experiments.admission import (
    DECLARATION,
    conditional_item_bootstrap,
    gate_review,
    planning,
    stratified_indices,
    summarize,
    views,
)
from third_eye.experiments.mechanism import ROLES, choose, folds


SEEDS = (12142, 12143, 12144, 12145)
OFFSETS = (50_000_000, 100_000_000)
CONTEXT = {"date_string": "11 Oct 2026"}
BUDGET_SU = 192.0
OVERHEAD_SU = (
    1.8525  # validation .11, prepare .33, CUDA 1, controller .0825, review .33
)


def runtime_limits(measured_seconds, budget=BUDGET_SU):
    """Conservative maxima within stream, 25% margin, quarter-hour ceilings.

    Inputs are full observed five-draw runtimes minus measured harvest/training
    time of the two omitted draws. Their evaluation/load overhead is retained.
    This is a planning guard, not a runtime guarantee for new parents.
    """
    if len(measured_seconds) != 4 or any(
        len(v) != 2 for v in measured_seconds.values()
    ):
        raise ValueError("Two measured pilots for each of four streams required")
    limits = {}
    for stream, values in measured_seconds.items():
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Positive finite measured runtimes required")
        limits[stream] = int(np.ceil(max(values) * 1.25 / 900) * 900)
    reservation = sum(limits.values()) * 4 / 3600 + OVERHEAD_SU
    if reservation > budget + 1e-9:
        raise ValueError(
            f"Measured pilot reservation {reservation:.4f} exceeds B cap {budget}"
        )
    return limits, reservation


def native_arrays(parent, immediate, candidates):
    if (
        len(immediate) != 3
        or len(candidates) != 3
        or any(len(v) != 3 for v in candidates)
    ):
        raise ValueError("Exactly three native draws for each K=3 candidate required")
    output = {}
    for role in ROLES:
        identity = [(x["id"], x["prompt_sha256"]) for x in parent["items"][role]]
        if len(identity) < 8 or len(set(identity)) != len(identity):
            raise ValueError("Unique paired items required")

        def values(artifact):
            items = artifact["items"][role]
            if [(x["id"], x["prompt_sha256"]) for x in items] != identity or any(
                type(x["correct"]) is not int or x["correct"] not in (0, 1)
                for x in items
            ):
                raise ValueError("Paired binary outcomes required")
            return np.asarray([x["correct"] for x in items], float)

        baseline = values(parent)
        output[role] = {
            "h1": np.stack([values(v) - baseline for v in immediate]),
            "native": np.stack(
                [
                    np.stack([values(candidates[k][s]) - baseline for k in range(3)])
                    for s in range(3)
                ]
            ),
        }
    return output


def native_summary(state):
    assignment, rows = folds(state["parent"]), []
    for fold in (0, 1):
        h1, selected, _ = views(state["panel"], assignment, fold)
        _, scored, _ = views(state["panel"], assignment, 1 - fold)
        greedy = choose(h1.mean(1))
        replay = (h1[:, 0] > 0) & (h1[:, 1] >= 0) & (h1[:, 2] >= 0)
        for held in range(3):
            admission = selected[[s for s in range(3) if s != held]].mean((0, 2)) > 0
            outcome = scored[held].mean(1)
            forced = float(greedy @ outcome)
            proxy = float((greedy * admission) @ outcome)
            rows.append(
                {
                    "forced_minus_hold": forced,
                    "proxy_minus_hold": proxy,
                    "proxy_minus_forced": proxy - forced,
                    "replay_minus_hold": float((greedy * replay) @ outcome),
                    "forced_retention": float(greedy @ scored[held, :, 2]),
                    "replay_retention": float((greedy * replay) @ scored[held, :, 2]),
                    "proxy_coverage": float(greedy @ admission),
                    "replay_coverage": float(greedy @ replay),
                }
            )
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def review(states):
    """Development evaluation only. Confirmation always requires human review."""
    streams = [s["stream"] for s in states]
    indices = stratified_indices(streams, parents_per_stream=4)
    fits = gate_review(states, expected_parents=16, validation_stride=4)
    summaries = [native_summary(s) for s in states]
    metrics = {k: summarize([s[k] for s in summaries], indices) for k in summaries[0]}
    proxy = np.asarray([s["proxy_minus_hold"] for s in summaries])
    forced = np.asarray([s["forced_minus_hold"] for s in summaries])
    headroom_samples = proxy[indices].mean(1) - np.maximum(0, forced[indices].mean(1))
    headroom = {
        "mean": float(proxy.mean() - max(0, forced.mean())),
        "ci95": np.quantile(headroom_samples, [0.025, 0.975]).tolist(),
        "privileged_proxy_not_oracle_or_upper_bound": True,
    }
    learned, power = {}, {}
    for horizon in ("h1", "h2"):
        parents = []
        for i in range(16):
            rows = [
                r["metrics"]
                for r in fits
                if r["test_index"] == i and r["horizon"] == horizon
            ]
            parents.append({k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
        learned[horizon] = {
            k: summarize([p[k] for p in parents], indices) for k in parents[0]
        }
        effects = [[p["minus_forced"], p["minus_hold"]] for p in parents]
        power[horizon] = {
            "overall_parent_sensitivity": planning(effects),
            "fixed_stream_residual_sensitivity": planning(effects, strata=streams),
        }
    # Frozen development criterion; H2 needs >0.5 pp more CV utility to displace H1.
    chosen = (
        "h2"
        if learned["h2"]["minus_hold"]["mean"] - learned["h1"]["minus_hold"]["mean"]
        > 0.005
        else "h1"
    )
    per_stream = {}
    for stream in sorted(set(streams)):
        group = [i for i, v in enumerate(streams) if v == stream]
        per_stream[stream] = {
            "parents": 4,
            "descriptive_only": True,
            **{
                k: float(np.mean([summaries[i][k] for i in group]))
                for k in summaries[0]
            },
        }
    return {
        "status": "development_review_required",
        "independent_parents": 16,
        "development_only": True,
        "automatic_expansion": False,
        "pooling_arm": False,
        "metrics": metrics,
        "proxy_minus_better_constant": headroom,
        "learned_gates": learned,
        "provisional_primary_horizon": chosen,
        "primary_selection_rule": "H1 unless H2 cross-validated utility advantage exceeds 0.5 pp; development only",
        "conditional_shared_item_uncertainty": conditional_item_bootstrap(states, fits),
        "power_planning_sensitivity": power,
        "streams": per_stream,
        "decision": "STOP_BEFORE_C"
        if headroom["mean"] < DECLARATION["practical_gain"]
        else "REVIEW_HEADROOM_POLICY_POWER_AND_COST_BEFORE_C",
        "limitations": "No population superiority or restored forecasting claim. Shared-item and parent intervals are conditional, not simultaneous joint confirmation intervals. Refit uncertainty and future-item generalization remain unresolved. Positive-part oracle estimates are forbidden.",
    }, fits
