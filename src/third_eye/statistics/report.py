"""Evidence-backed research reports and conservative scaling decisions."""

from collections import defaultdict
import numpy as np

from third_eye.forecasting.encoding import HEADS, labels
from third_eye.statistics.inference import bootstrap


def phenomenon(records):
    states = defaultdict(list)
    for row in records:
        states[row["state_id"]].append(row)
    rows = []
    for state, candidates in states.items():
        h1 = np.asarray([r["labels"]["utility_h1"] for r in candidates])
        h2 = np.asarray([r["labels"]["utility_h2"] for r in candidates])
        h1, h2 = np.round(h1, 12), np.round(h2, 12)
        immediate = set(np.flatnonzero(h1 == h1.max()))
        future = set(np.flatnonzero(h2 == h2.max()))
        harm = [any(r["labels"]["h1"][h] < 0 for h in HEADS) for r in candidates]
        rows.append(
            {
                "state_id": state,
                "trajectory_id": candidates[0]["trajectory_id"],
                "model": candidates[0]["model"],
                "generation": candidates[0]["generation"],
                "strict_ranking_reversal": not bool(immediate & future),
                "unique_winners": len(immediate) == len(future) == 1,
                "harmful_component_fraction": float(np.mean(harm)),
                "harmful_utility_fraction": float(np.mean(h1 < 0)),
                "catastrophic_utility_fraction": float(np.mean(h1 < -0.02)),
                "target_gain_retention_loss_fraction": float(
                    np.mean(
                        [
                            r["labels"]["h2"]["target"] > 0
                            and r["labels"]["h2"]["retention"] < 0
                            for r in candidates
                        ]
                    )
                ),
            }
        )
    # State-level bootstrap for ranking; trajectory clustering additionally
    # respects correlations between successive states of an exploration run.
    trajectories = defaultdict(list)
    for row in rows:
        trajectories[row["trajectory_id"]].append(row)
    result = {"states": len(rows), "trajectories": len(trajectories), "per_state": rows}
    terminal = sum(r.get("continuation_available", 1) == 0 for r in records)
    result["correction_scarcity"] = {
        "terminal_continuations": terminal,
        "candidate_branches": len(records),
        "terminal_fraction": terminal / len(records) if records else 0.0,
        "interpretation": "Descriptive availability; not a saturation claim",
    }
    for key in (
        "strict_ranking_reversal",
        "harmful_component_fraction",
        "harmful_utility_fraction",
        "catastrophic_utility_fraction",
        "target_gain_retention_loss_fraction",
    ):
        result[key] = {
            "state_bootstrap": bootstrap([r[key] for r in rows]),
            "trajectory_bootstrap": bootstrap(
                [np.mean([r[key] for r in group]) for group in trajectories.values()]
            ),
        }
    return result


def gate1(report, min_states=20):
    mismatch = report["strict_ranking_reversal"]["state_bootstrap"]["mean"]
    harm = report["harmful_component_fraction"]["state_bootstrap"]["mean"]
    return {
        "gate": 1,
        "passed": report["states"] >= min_states and (mismatch >= 0.20 or harm >= 0.15),
        "sample_sufficient": report["states"] >= min_states,
        "status": "insufficient_evidence"
        if report["states"] < min_states
        else (
            "passed" if mismatch >= 0.20 or harm >= 0.15 else "phenomenon_not_observed"
        ),
        "minimum_states": min_states,
        "states": report["states"],
        "mismatch_fraction": mismatch,
        "harmful_component_fraction": harm,
        "thresholds": {"mismatch": 0.20, "harmful_components": 0.15},
    }


def gate2(metrics, kind, horizon, min_states=30):
    rho = metrics["ranking"]["spearman"]
    accuracy = metrics["ranking"].get("informative_top1", metrics["ranking"]["top1"])
    chance = metrics["ranking"].get("informative_chance_top1", 1 / 3)
    useful = metrics["nonconstant_states"] >= min_states
    return {
        "gate": 2,
        "passed": kind == "direct"
        and horizon == 2
        and useful
        and rho is not None
        and rho >= 0.30
        and accuracy is not None
        and accuracy >= 0.45
        and accuracy > chance,
        "sample_sufficient": useful,
        "minimum_nonconstant_states": min_states,
        "states": metrics["states"],
        "nonconstant_states": metrics["nonconstant_states"],
        "spearman": rho,
        "top1": accuracy,
        "tie_adjusted_chance_top1": chance,
        "kind": kind,
        "horizon": horizon,
        "evaluation_partition": "validation",
        "thresholds": {"spearman": 0.30, "top1": 0.45},
    }


def trajectory_metrics(run, weights=(1 / 3, 1 / 3, 1 / 3)):
    if not run.get("complete"):
        raise ValueError("Incomplete trajectories cannot enter policy comparisons")
    points = run["points"]
    if len(points) < 2:
        raise ValueError("Trajectory needs a baseline and a final point")
    utility = np.asarray([p["utility"] for p in points])
    retention = np.asarray([p["scores"]["retention"] for p in points])
    improvement = utility - utility[0]
    times = sum(p.get("seconds", 0) for p in points)
    delta = np.diff(utility)
    return {
        "final_gain": float(improvement[-1]),
        "auc_gain": float(np.trapezoid(improvement)),
        "best_gain": float(improvement.max()),
        "harmful_update_rate": float(np.mean(delta < 0)),
        "catastrophic_regression_rate": float(np.mean(delta < -0.02)),
        "retention_delta": float(retention[-1] - retention[0]),
        "maximum_forgetting": float((retention[0] - retention).max()),
        "wall_gpu_hours": times / 3600,
        "gain_per_gpu_hour": float(improvement[-1] / (times / 3600)) if times else None,
        "generations": len(points) - 1,
    }


def forecast_report(records, forecaster, horizon=2):
    from third_eye.forecasting.training import evaluate_predictions

    training = set(forecaster.metadata.get("selection_state_ids", []))
    trajectories = set(forecaster.metadata.get("selection_trajectory_ids", []))
    if any(
        r["state_id"] in training or r["trajectory_id"] in trajectories for r in records
    ):
        raise ValueError(
            "Reported forecast test data overlaps training/validation selection"
        )
    prediction = forecaster.predict(records)
    result = evaluate_predictions(
        records, prediction, forecaster.metadata["utility_weights"], horizon
    )
    ranking = result["per_state"]
    for metric in ("spearman", "kendall", "ndcg_at_3", "top1"):
        values = [r[metric] for r in ranking.values() if r[metric] is not None]
        result.setdefault("state_ci95", {})[metric] = (
            bootstrap(values) if values else None
        )
    truth = np.stack([labels(r, horizon) for r in records])
    result["predictions"] = [
        {
            "state_id": r["state_id"],
            "candidate_id": r["candidate_id"],
            "truth": truth[i].tolist(),
            "prediction": prediction[i].tolist(),
        }
        for i, r in enumerate(records)
    ]
    return result
