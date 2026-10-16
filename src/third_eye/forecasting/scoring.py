"""Independently scored selector values with trajectory-level uncertainty."""

from collections import defaultdict

import numpy as np

from third_eye.forecasting.continuation import repeated_targets, state_groups
from third_eye.forecasting.encoding import HEADS
from third_eye.statistics.inference import paired_comparison, holm


def _winner_value(scores, values):
    tied = np.isclose(scores, max(scores), atol=1e-12, rtol=0)
    return float(values[tied].mean())


def independent_selection_report(rows, decisions, weights):
    """Never choose an oracle from the outcomes being scored.

    Primary: forced learned future selection versus observed immediate choice.
    Report abstention separately; holding parameters is zero under this H2
    reference. Confidence intervals cluster whole trajectories, not K rows.
    """
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (3,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Invalid utility weights")
    mean, _, _ = repeated_targets(rows)
    utility = mean @ weights
    states = []
    for indices in state_groups(rows):
        group = [rows[i] for i in indices]
        state = group[0]["state_id"]
        decision = decisions[state]
        if decision["candidate_ids"] != [r["candidate_id"] for r in group]:
            raise ValueError("Decision candidate order differs from scored outcomes")
        for r in group:
            if (
                not r.get("selection_items")
                or not r.get("scoring_items")
                or set(r["selection_items"]) & set(r["scoring_items"])
            ):
                raise ValueError(
                    "Selection and scoring items must be nonempty and disjoint"
                )
            if set(r["selection_seeds"]) & {
                s["seed"] for s in r["continuation_replicates"]
            }:
                raise ValueError("Selection and scoring continuation seeds overlap")
        values = utility[indices]
        immediate = (
            np.asarray([[r["selection_h1"][h] for h in HEADS] for r in group]) @ weights
        )
        means = np.asarray(decision["utility_mean"], dtype=float)
        scores = np.asarray(decision["risk_score"], dtype=float)
        predicted_h1 = np.asarray(decision["h1"], dtype=float) @ weights
        if any(
            a.shape != (3,) or not np.isfinite(a).all()
            for a in (immediate, means, scores, predicted_h1)
        ):
            raise ValueError("Invalid selector scores")
        selected = decision["selected"]
        if selected is not None and (
            type(selected) is not int or not 0 <= selected < 3
        ):
            raise ValueError("Invalid selector action")
        if bool(decision["abstained"]) != (selected is None):
            raise ValueError("Selector abstention disagrees with its action")
        states.append(
            {
                "state_id": state,
                "trajectory_id": group[0]["trajectory_id"],
                "forecast_mean_only": _winner_value(means, values),
                "forecast_risk_forced": _winner_value(scores, values),
                "forecast_policy": 0.0 if selected is None else float(values[selected]),
                "predicted_h1": _winner_value(predicted_h1, values),
                "observed_h1": _winner_value(immediate, values),
                "random": float(values.mean()),
                "no_update": 0.0,
                "abstained": bool(selected is None),
            }
        )
    if set(decisions) != {r["state_id"] for r in states}:
        raise ValueError("Decision and scoring state coverage differs")
    grouped = defaultdict(list)
    for r in states:
        grouped[r["trajectory_id"]].append(r)
    if len(grouped) < 2:
        raise ValueError("Independent scoring requires >=2 independent trajectories")
    comparisons = {}
    for method, baseline in (
        ("forecast_mean_only", "observed_h1"),
        ("forecast_mean_only", "predicted_h1"),
        ("forecast_mean_only", "random"),
        ("forecast_policy", "observed_h1"),
        ("forecast_policy", "no_update"),
        ("forecast_policy", "forecast_risk_forced"),
    ):
        comparisons[f"{method}_minus_{baseline}"] = paired_comparison(
            [np.mean([r[method] for r in group]) for group in grouped.values()],
            [np.mean([r[baseline] for r in group]) for group in grouped.values()],
        )
    primary = "forecast_mean_only_minus_observed_h1"
    secondary = holm(
        {k: v["paired_permutation_p"] for k, v in comparisons.items() if k != primary}
    )
    for name, p in secondary.items():
        comparisons[name]["holm_adjusted_p"] = p
    return {
        "primary_comparison": primary,
        "comparisons": comparisons,
        "independent_units": "whole trajectories, equal trajectory weighting",
        "trajectories": len(grouped),
        "states": len(states),
        "per_state": states,
        "interpretation": "Conditional on disjoint scoring items and fresh continuation seeds. Mean-only selection is primary; risk/abstention gains are separate. Intervals do not include sampling new evaluation item pools. No noisy measured-oracle regret is reported.",
    }
