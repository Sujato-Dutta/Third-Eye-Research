"""Development-only whole-trajectory forecasting and held-out-draw diagnostics."""

import copy
from itertools import combinations

import numpy as np

from third_eye.forecasting.continuation import (
    candidate_inputs,
    reference_h1,
    repeated_targets,
    state_groups,
)
from third_eye.forecasting.encoding import HEADS, FEATURE_NAMES
from third_eye.forecasting.training import train_forecaster
from third_eye.io import digest, write_json
from third_eye.statistics.inference import paired_comparison, holm
from third_eye.statistics.metrics import ranking


SEEDS = (42, 1042, 2042)
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def pilot_folds(records):
    """Leave one whole trajectory out; fixed distinct-stream validation partner."""
    groups = state_groups(records)
    if len(groups) != 8 or len({r["trajectory_id"] for r in records}) != 8:
        raise ValueError("Review requires eight one-state pilot trajectories")
    for test_index in range(8):
        validation_index = (test_index + 2) % 8
        parts = {"train": [], "validation": [], "test": []}
        for i, indices in enumerate(groups):
            role = (
                "test"
                if i == test_index
                else "validation"
                if i == validation_index
                else "train"
            )
            parts[role].extend(records[j] for j in indices)
        yield parts


def mean_views(rows):
    """Temporary regression targets; native labels remain unchanged."""
    mean, _, _ = repeated_targets(rows, minimum_replicates=5)
    result = copy.deepcopy(rows)
    for row, value in zip(result, mean):
        row["labels"]["h2"] = dict(zip(HEADS, value.tolist()))
    return result


def simple_inputs(rows):
    values = candidate_inputs(rows)
    n = len(FEATURE_NAMES)
    selected = [
        FEATURE_NAMES.index(name)
        for name in (
            "pre_update_nll",
            "probe_candidate_loss_change",
            "probe_anchor_loss_change",
            "retention_gradient_cosine",
        )
    ]
    columns = list(range(3))
    for offset in (3, 3 + n, 3 + 2 * n):
        columns.extend(offset + i for i in selected)
    return values[:, columns]


def ridge_prediction(parts, horizon=2, simple=False):
    """Normalize on training only; choose regularization on validation only."""
    encoder = simple_inputs if simple else candidate_inputs
    arrays = {role: encoder(rows).astype(float) for role, rows in parts.items()}
    center, scale = arrays["train"].mean(0), arrays["train"].std(0)
    scale[scale < 1e-6] = 1
    x = {role: np.clip((a - center) / scale, -20, 20) for role, a in arrays.items()}

    def target(rows):
        return (
            repeated_targets(rows)[0]
            if horizon == 2
            else np.stack([reference_h1(r) for r in rows])
        )

    y, validation_y = target(parts["train"]), target(parts["validation"])
    intercept = y.mean(0)
    best = None
    for alpha in ALPHAS:
        # Dual solve is small even with the full feature allowlist and masks.
        dual = np.linalg.solve(
            x["train"] @ x["train"].T + alpha * np.eye(len(y)), y - intercept
        )
        coef = x["train"].T @ dual
        error = float(np.mean((x["validation"] @ coef + intercept - validation_y) ** 2))
        if best is None or error < best[0]:
            best = error, alpha, coef
    return x["test"] @ best[2] + intercept, {
        "alpha": best[1],
        "validation_mse": best[0],
        "training_center": center.tolist(),
        "training_scale": scale.tolist(),
    }


def tied_value(scores, values):
    scores, values = np.asarray(scores), np.asarray(values)
    selected = np.isclose(scores, scores.max(), atol=1e-12, rtol=0)
    return float(values[selected].mean())


def held_out_draws(rows, weights):
    """Select on three draws, score on two others; shared evaluation items remain."""
    _, _, draws = repeated_targets(rows, minimum_replicates=5)
    utilities = np.stack(draws) @ weights
    if utilities.shape != (3, 5):
        raise ValueError(
            "Draw comparison requires one K=3 state and exactly five draws"
        )
    h1 = np.stack([reference_h1(r) for r in rows]) @ weights
    selections = []
    for selected in combinations(range(5), 3):
        scoring = tuple(i for i in range(5) if i not in selected)
        score = utilities[:, scoring].mean(1)
        selectors = utilities[:, selected].mean(1)
        selections.append(
            {
                "selection_draw_indices": list(selected),
                "scoring_draw_indices": list(scoring),
                "continuation_selected": tied_value(selectors, score),
                "observed_h1": tied_value(h1, score),
                "random": float(score.mean()),
            }
        )
    return {
        "state_id": rows[0]["state_id"],
        "trajectory_id": rows[0]["trajectory_id"],
        "rotations": selections,
        "mean_selected_minus_h1": float(
            np.mean([s["continuation_selected"] - s["observed_h1"] for s in selections])
        ),
        "interpretation": "Privileged draw-selection diagnostic, not a learned forecaster. Disjoint continuation draws share evaluation items. Rotations are not independent samples.",
    }


def check_frozen_model(records, model):
    if model.metadata["kind"] != "expected" or not np.allclose(
        model.metadata["utility_weights"], records[0]["utility_weights"]
    ):
        raise ValueError("Frozen forecast kind/utility differs")
    if any(
        r["trajectory_id"] in model.metadata["selection_trajectory_ids"]
        or r["state_id"] in model.metadata["selection_state_ids"]
        for r in records
    ):
        raise ValueError(
            "Frozen forecast was developed on an evaluation state/trajectory"
        )


def run_review(records, output, streams, epochs=100, seeds=SEEDS, frozen_model=None):
    """All outcome-dependent computation belongs on an allocated CPU node."""
    import torch

    torch.set_num_threads(4)
    if frozen_model is not None:
        check_frozen_model(records, frozen_model)
    output.mkdir(parents=True, exist_ok=False)
    primary = (
        "frozen_expected_minus_observed_h1"
        if frozen_model is not None
        else "expected_minus_matched_h1"
    )
    write_json(
        output / "declaration.json",
        {
            "development_only": True,
            "epochs": epochs,
            "patience": 15,
            "hidden": 16,
            "model_seeds": list(seeds),
            "ridge_alphas": list(ALPHAS),
            "folds": 8,
            "primary": primary,
            "input_hash": digest(records),
            "larger_collection_released": False,
            "interpretation": "Previously inspected initial-state pilots; cross-validation is development evidence, not a frozen independent confirmation or Gate 2.",
        },
    )
    states, draw_review = [], []
    weights = np.asarray(records[0]["utility_weights"], dtype=float)
    if frozen_model is not None:
        views = [
            {k: r[k] for k in ("state_id", "candidate_id", "precommit")}
            for r in records
        ]
        fixed = frozen_model.predict_distribution(views)
        fixed_by_state = {}
        for indices in state_groups(records):
            fixed_by_state[records[indices[0]]["state_id"]] = {
                k: v[indices] for k, v in fixed.items()
            }
        write_json(
            output / "frozen_predictions.json",
            {
                "weights_sha256": frozen_model.metadata["weights_sha256"],
                "metadata_hash": digest(frozen_model.metadata),
                "predictions": fixed["mean"].tolist(),
                "candidate_order": [
                    (r["state_id"], r["candidate_id"]) for r in records
                ],
                "interpretation": "Checkpoint fixed before pilot collection; evaluation/report design is retrospective. Shared evaluation items remain.",
            },
        )
    for fold, parts in enumerate(pilot_folds(records)):
        target, _, _ = repeated_targets(parts["test"])
        native = {role: mean_views(rows) for role, rows in parts.items()}
        predictions, distributions = {}, []
        seed_predictions = {}
        for kind, horizon in (("expected", 2), ("matched_h1", 1), ("direct", 2)):
            runs = []
            for seed in seeds:
                model = train_forecaster(
                    parts if kind == "expected" else native,
                    output / f"fold_{fold}" / f"{kind}_{seed}",
                    kind=kind,
                    horizon=horizon,
                    weights=weights.tolist(),
                    hidden=16,
                    epochs=epochs,
                    patience=15,
                    seed=seed,
                )
                views = [
                    {k: r[k] for k in ("state_id", "candidate_id", "precommit")}
                    for r in parts["test"]
                ]
                forecast = model.predict(views)
                runs.append(forecast)
                if kind == "expected":
                    distributions.append(model.predict_distribution(views))
            predictions[kind] = np.mean(runs, axis=0)
            seed_predictions[kind] = [p.tolist() for p in runs]
        predictions["expected_h1_component"] = np.mean(
            [d["h1"] for d in distributions], axis=0
        )
        if frozen_model is not None:
            frozen = fixed_by_state[parts["test"][0]["state_id"]]
            predictions["frozen_expected"] = frozen["mean"]
            predictions["frozen_h1_component"] = frozen["h1"]
        for name, horizon, simple in (
            ("linear_h2", 2, False),
            ("simple_h2", 2, True),
            ("linear_h1", 1, False),
        ):
            predictions[name], receipt = ridge_prediction(parts, horizon, simple)
            write_json(output / f"fold_{fold}" / f"{name}.json", receipt)
        truth = target @ weights
        h1 = np.stack([reference_h1(r) for r in parts["test"]]) @ weights
        values = {
            method: tied_value(p @ weights, truth) for method, p in predictions.items()
        }
        values.update(
            observed_h1=tied_value(h1, truth), random=float(truth.mean()), no_update=0.0
        )
        means = predictions["expected"] @ weights
        spread = np.sqrt(
            np.mean([d["utility_std"] ** 2 for d in distributions], axis=0)
        )
        values["expected_abstain"] = 0.0 if means.max() <= 0 else values["expected"]
        risk = (
            means - spread
        )  # Fixed beta=1 development diagnostic, no threshold tuning.
        values["expected_risk_abstain"] = (
            0.0 if risk.max() <= 0 else tied_value(risk, truth)
        )
        scores = {
            method: ranking(truth, p @ weights) for method, p in predictions.items()
        }
        scores["observed_h1"] = ranking(truth, h1)
        row = parts["test"][0]
        state = {
            "state_id": row["state_id"],
            "trajectory_id": row["trajectory_id"],
            "stream": streams[row["state_id"]],
            "values": values,
            "rankings": scores,
            "truth_mean_h2": target.tolist(),
            "predictions": {k: v.tolist() for k, v in predictions.items()},
            "seed_predictions": seed_predictions,
            "training_trajectories": sorted(
                {r["trajectory_id"] for r in parts["train"]}
            ),
            "validation_trajectories": sorted(
                {r["trajectory_id"] for r in parts["validation"]}
            ),
        }
        states.append(state)
        draw_review.append(held_out_draws(parts["test"], weights))
        write_json(output / f"fold_{fold}" / "out_of_fold.json", state)
    contrasts = [
        ("expected", b)
        for b in (
            "matched_h1",
            "direct",
            "linear_h2",
            "simple_h2",
            "observed_h1",
            "random",
            "no_update",
        )
    ]
    contrasts += [
        ("expected_abstain", "no_update"),
        ("expected_risk_abstain", "no_update"),
    ]
    contrasts += [("expected", "expected_h1_component")]
    if frozen_model is not None:
        contrasts += [
            ("frozen_expected", b)
            for b in ("observed_h1", "random", "no_update", "frozen_h1_component")
        ]
    comparisons = {
        f"{a}_minus_{b}": paired_comparison(
            [s["values"][a] for s in states], [s["values"][b] for s in states]
        )
        for a, b in contrasts
    }
    adjusted = holm(
        {k: v["paired_permutation_p"] for k, v in comparisons.items() if k != primary}
    )
    for key, value in adjusted.items():
        comparisons[key]["holm_adjusted_p"] = value
    ranking_summary = {}
    for method in states[0]["rankings"]:
        ranking_summary[method] = {}
        for metric in ("top1", "chance_top1", "spearman"):
            valid = [
                s["rankings"][method][metric]
                for s in states
                if s["rankings"][method][metric] is not None
            ]
            ranking_summary[method][metric] = float(np.mean(valid)) if valid else None
    report = {
        "development_only": True,
        "gate2_passed": None,
        "automatic_expansion": False,
        "states": states,
        "trajectories": 8,
        "primary_comparison": primary,
        "comparisons": comparisons,
        "ranking_summary": ranking_summary,
        "held_out_draw_diagnostic": draw_review,
        "limitations": [
            "Eight previously inspected initial-state trajectories, two per stream.",
            "Cross-validation training sets overlap; resampling intervals are descriptive and conditional on fitted predictions.",
            "Shared finite evaluation items; held-out draws do not replace independent scoring items.",
            "Uniform averaging over three model seeds; seeds and draw rotations are not additional independent units.",
            "A favorable result cannot establish recursive improvement, superiority on fresh cohorts or Gate 2.",
            "Expected H1 components share H2-trained representations; they are decomposition diagnostics, not independently H1-only trained controls.",
            "Matched_h1 is the earlier temporal H1 baseline; its architecture matches Direct, not the expected-value MLP.",
        ],
    }
    write_json(output / "report.json", report)
    return report
