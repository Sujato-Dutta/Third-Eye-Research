"""Exploratory admission diagnostics with independent item/draw scoring."""

from itertools import combinations
from statistics import NormalDist

import numpy as np

from third_eye.experiments.crossed import decompose
from third_eye.experiments.mechanism import ROLES, choose, folds


DECLARATION = {
    "version": "admission_stage_a_20261018",
    "development_only": True,
    "previously_inspected_data": True,
    "automatic_expansion": False,
    "independent_units": "eight_parent_trajectories_two_per_stream",
    "item_folds": "third-eye-lr-mechanism-20261013",
    "draws": 3,
    "tie_policy": "uniform_maximum_rounded_12_decimals; zero_gain_holds",
    "proxy": "select_admission_on_two_draws_one_item_half_score_other_draw_half",
    "features": ["h1_target", "h1_ood", "h1_retention", "log1p_effective_update_norm"],
    "excluded_features": "historical_text_probes_without_matched_rendering_context",
    "ridge_alphas": [0.1, 1.0, 10.0, 100.0],
    "admission_thresholds": [0.0, 0.0025, 0.005, 0.01],
    "retention_prediction_floor": -0.005,
    "cv": "leave_one_parent_out_validation_index_plus_two_mod_eight",
    "training_items": "selection_half_only_including_training_and_validation_targets",
    "scoring_items": "other_half_never_used_for_model_or_threshold_fitting",
    "replay_gate": "fixed_immediate_candidate_target_positive_ood_and_retention_nonnegative",
    "item_fractions": [0.25, 0.5, 1.0],
    "candidate_subsets": "all_subsets_K1_K2_K3_averaged_within_parent",
    "bootstrap": "5000_stream_stratified_parent_resamples_conditional_on_CV_fits",
    "power": "normal_planning_sensitivity_not_confirmed_power",
    "practical_gain": 0.005,
    "norm": "Frobenius_of_scaled_BA_candidate_minus_parent_no_SVD",
    "legacy_training_exclusion": "off_diagonal_never_train; native_diagonal_exploratory_admission_view_only_authorized_by_research_plan; source_flags_unchanged",
    "optimizer_and_harvest_variances": "not_separately_identified_in_fresh_draws",
}


def effective_delta_norm(parent, candidate, scale):
    """Gauge-invariant norm using small Gram products; no dense update or SVD.

    Inputs map layer names to (A, B), with equal layer sets. LoRA factors may
    use different gauges. This is the actual update, not a short-probe norm.
    """
    if (
        not np.isfinite(scale)
        or scale <= 0
        or not parent
        or parent.keys() != candidate.keys()
    ):
        raise ValueError("Matched LoRA layers and positive finite scale required")
    squared = 0.0
    for name in parent:
        a0, b0 = [np.asarray(x, dtype=np.float64) for x in parent[name]]
        a1, b1 = [np.asarray(x, dtype=np.float64) for x in candidate[name]]
        if (
            any(x.ndim != 2 or not np.isfinite(x).all() for x in (a0, b0, a1, b1))
            or a0.shape != a1.shape
            or b0.shape != b1.shape
            or a0.shape[0] != b0.shape[1]
        ):
            raise ValueError("Finite compatible adapter factors required")
        left = np.concatenate((b1, -b0), axis=1)
        right = np.concatenate((a1, a0), axis=0)
        value = float(np.sum((left.T @ left) * (right @ right.T)))
        tolerance = 1e-10 * max(
            1.0, np.linalg.norm(left) ** 2 * np.linalg.norm(right) ** 2
        )
        if value < -tolerance:
            raise ValueError(
                "Negative effective squared norm beyond numerical tolerance"
            )
        squared += max(0.0, value) * scale**2
    return float(np.sqrt(squared))


def arrays(parent, immediate, cube):
    """Align binary outcomes to the parent, returning item-level paired deltas.

    Cube order is recipient, donor, draw. Array order is draw, recipient, donor,
    item. The donor counterfactuals never enter a learned gate target.
    """
    if (
        len(immediate) != 3
        or len(cube) != 3
        or any(len(row) != 3 or any(len(draws) != 3 for draws in row) for row in cube)
    ):
        raise ValueError("Complete K3 by K3 by three-draw panel required")
    output = {}
    for role in ROLES:
        identity = [(x["id"], x["prompt_sha256"]) for x in parent["items"][role]]
        if len(identity) < 8 or len(set(identity)) != len(identity):
            raise ValueError("At least eight unique items per role required")

        def values(artifact):
            items = artifact["items"][role]
            if [(x["id"], x["prompt_sha256"]) for x in items] != identity or any(
                type(x["correct"]) is not int or x["correct"] not in (0, 1)
                for x in items
            ):
                raise ValueError("Paired binary item identities/order required")
            return np.asarray([x["correct"] for x in items], dtype=float)

        baseline = values(parent)
        output[role] = {
            "h1": np.asarray([values(a) - baseline for a in immediate]),
            "cube": np.asarray(
                [
                    [
                        [values(cube[i][j][s]) - baseline for j in range(3)]
                        for i in range(3)
                    ]
                    for s in range(3)
                ]
            ),
        }
    return output


def views(panel, assignment, fold, fraction=1.0):
    h1, future, crossed = [], [], []
    for role in ROLES:
        index = assignment[role][fold]
        index = index[: max(1, int(len(index) * fraction))]
        h1.append(panel[role]["h1"][:, index].mean(-1))
        if "native" in panel[role]:
            future.append(panel[role]["native"][:, :, index].mean(-1))
        else:
            x = panel[role]["cube"][:, :, :, index].mean(-1)
            crossed.append(x)
            future.append(x[:, np.arange(3), np.arange(3)])
    return (
        np.stack(h1, -1),
        np.stack(future, -1),
        np.stack(crossed, -1) if crossed else None,
    )


def state_summary(parent, panel):
    """Admission actions use selection data; scoring is signed and never clipped."""
    assignment = folds(parent)
    rows, optimism = [], {str(k): [] for k in (1, 2, 3)}
    crossdraw_gaps = {str(k): [] for k in (1, 2, 3)}
    agreement, head_agreement = [], []
    observed_variation = []
    fractions = {str(f): [] for f in DECLARATION["item_fractions"]}
    for fold in (0, 1):
        h1, future, cross = views(panel, assignment, fold)
        _, heldout, held_cross = views(panel, assignment, 1 - fold)
        greedy = choose(h1.mean(1))
        replay = (h1[:, 0] > 0) & (h1[:, 1] >= 0) & (h1[:, 2] >= 0)
        for score_draw in range(3):
            select_draws = [s for s in range(3) if s != score_draw]
            ahead = future[select_draws].mean(0).mean(1)
            score = heldout[score_draw].mean(1)
            admission = ahead > 0
            forced = float(greedy @ score)
            proxy = float((greedy * admission) @ score)
            rows.append(
                {
                    "forced_minus_hold": forced,
                    "proxy_minus_hold": proxy,
                    "proxy_minus_forced": proxy - forced,
                    "proxy_admission_probability": float(greedy @ admission),
                    "replay_minus_hold": float((greedy * replay) @ score),
                    "replay_minus_forced": float((greedy * replay - greedy) @ score),
                    "forced_retention": float(greedy @ heldout[score_draw, :, 2]),
                    "replay_retention": float(
                        (greedy * replay) @ heldout[score_draw, :, 2]
                    ),
                    "replay_admission_probability": float(greedy @ replay),
                    "noisy_positive_part_minus_independent_admission": float(
                        greedy @ np.maximum(score, 0) - proxy
                    ),
                }
            )
            residual = decompose(cross[select_draws].mean((0, 3)))["interaction"]
            scored_residual = decompose(held_cross[score_draw].mean(-1))["interaction"]
            agreement.append(float(np.mean(np.asarray(residual) * scored_residual)))
            head_agreement.append(
                [
                    float(
                        np.mean(
                            np.asarray(
                                decompose(cross[select_draws, :, :, h].mean(0))[
                                    "interaction"
                                ]
                            )
                            * decompose(held_cross[score_draw, :, :, h])["interaction"]
                        )
                    )
                    for h in range(3)
                ]
            )
            for size in (1, 2, 3):
                subset_values, item_values = [], []
                for subset in combinations(range(3), size):
                    indices = np.asarray(subset)
                    local_selection = np.round(ahead[indices], 12)
                    ahead_mass = (local_selection == local_selection.max()).astype(
                        float
                    )
                    ahead_mass /= ahead_mass.sum()
                    local_score = np.round(score[indices], 12)
                    noisy_mass = (local_score == local_score.max()).astype(float)
                    noisy_mass /= noisy_mass.sum()
                    subset_values.append(
                        float(noisy_mass @ score[indices] - ahead_mass @ score[indices])
                    )
                    same_draw = np.round(future[score_draw, indices].mean(1), 12)
                    item_mass = (same_draw == same_draw.max()).astype(float)
                    item_mass /= item_mass.sum()
                    item_values.append(float((noisy_mass - item_mass) @ score[indices]))
                optimism[str(size)].append(float(np.mean(item_values)))
                crossdraw_gaps[str(size)].append(float(np.mean(subset_values)))
            for fraction in DECLARATION["item_fractions"]:
                small_h1, small_future, _ = views(panel, assignment, fold, fraction)
                _, small_scored, _ = views(panel, assignment, 1 - fold, fraction)
                selected = choose(small_future[select_draws].mean((0, 2)))
                noisy = choose(small_scored[score_draw].mean(1))
                values = small_scored[score_draw].mean(1)
                fractions[str(fraction)].append(
                    {
                        "selection_optimism": float((noisy - selected) @ values),
                        "immediate_selected_value": float(
                            choose(small_h1.mean(1)) @ values
                        ),
                    }
                )
        observed_variation.append(
            {
                "candidate_mean_variance": float(np.var(future.mean((0, 2)), ddof=1)),
                "within_candidate_draw_variance": float(
                    np.var(future.mean(2), axis=0, ddof=1).mean()
                ),
            }
        )
    item_variance = []
    for role in ROLES:
        diagonal = panel[role]["cube"][:, np.arange(3), np.arange(3)]
        item_variance.append(
            float(np.var(diagonal, axis=-1, ddof=1).mean() / diagonal.shape[-1])
        )
    return {
        "metrics": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
        "fold_draw_values": rows,
        "interaction_agreement": float(np.mean(agreement)),
        "interaction_agreement_by_head": np.mean(head_agreement, axis=0).tolist(),
        "same_outcome_selection_optimism_by_K": {
            k: float(np.mean(v)) for k, v in optimism.items()
        },
        "cross_draw_selection_gap_by_K": {
            k: float(np.mean(v)) for k, v in crossdraw_gaps.items()
        },
        "item_fraction_sensitivity": {
            k: {m: float(np.mean([r[m] for r in v])) for m in v[0]}
            for k, v in fractions.items()
        },
        "observed_variation": {
            **{
                k: float(np.mean([r[k] for r in observed_variation]))
                for k in observed_variation[0]
            },
            "conditional_paired_item_mean_variance_proxy": float(
                sum(item_variance) / 9
            ),
            "components_are_not_additive_or_noise_corrected": True,
            "harvest_optimizer_components_identified": False,
        },
    }


def ridge_fit(x, y, alpha):
    x, y = np.asarray(x, float), np.asarray(y, float)
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    z = (x - center) / scale
    intercept = y.mean(0)
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "intercept": intercept.tolist(),
        "coef": coef.tolist(),
    }


def ridge_predict(model, x):
    return ((np.asarray(x) - model["center"]) / model["scale"]) @ model["coef"] + model[
        "intercept"
    ]


def gate_mass(immediate, predicted, threshold):
    """Do not reselect the candidate using predictions or held-out outcomes."""
    return choose(np.asarray(immediate).mean(1)) * (
        (np.asarray(predicted).mean(1) > threshold)
        & (np.asarray(predicted)[:, 2] >= DECLARATION["retention_prediction_floor"])
    )


def gate_review(states, expected_parents=8, validation_stride=2):
    """Whole-parent nested fits; train/validation never access scoring items."""
    if (
        len(states) != expected_parents
        or len({s["trajectory_id"] for s in states}) != expected_parents
    ):
        raise ValueError("Declared distinct parent groups required")
    output = []
    for fold in (0, 1):
        prepared = []
        for state in states:
            assignment = folds(state["parent"])
            h1, future, _ = views(state["panel"], assignment, fold)
            _, scored, _ = views(state["panel"], assignment, 1 - fold)
            norms = np.asarray(state["effective_update_norms"], float)
            if norms.shape != (3,) or not np.isfinite(norms).all() or (norms < 0).any():
                raise ValueError("Finite nonnegative effective update norms required")
            prepared.append(
                {
                    "x": np.column_stack((h1, np.log1p(norms))),
                    "h1": h1,
                    "h2": future.mean(0),
                    "scored": scored.mean(0),
                }
            )
        for test in range(expected_parents):
            validation = (test + validation_stride) % expected_parents
            train = [i for i in range(expected_parents) if i not in (test, validation)]
            for horizon in ("h1", "h2"):
                x = np.concatenate([prepared[i]["x"] for i in train])
                y = np.concatenate([prepared[i][horizon] for i in train])
                candidates = []
                for alpha in DECLARATION["ridge_alphas"]:
                    model = ridge_fit(x, y, alpha)
                    prediction = ridge_predict(model, prepared[validation]["x"])
                    mse = float(
                        np.mean((prediction - prepared[validation][horizon]) ** 2)
                    )
                    candidates.append((mse, alpha, model, prediction))
                _, alpha, model, valid_prediction = min(
                    candidates, key=lambda row: (row[0], -row[1])
                )
                thresholds = []
                for threshold in DECLARATION["admission_thresholds"]:
                    mass = gate_mass(
                        prepared[validation]["h1"], valid_prediction, threshold
                    )
                    value = float(mass @ prepared[validation][horizon].mean(1))
                    thresholds.append((value, threshold))
                _, threshold = max(
                    thresholds, key=lambda row: (round(row[0], 12), row[1])
                )
                prediction = ridge_predict(model, prepared[test]["x"])
                mass = gate_mass(prepared[test]["h1"], prediction, threshold)
                greedy = choose(prepared[test]["h1"].mean(1))
                replay = (
                    (prepared[test]["h1"][:, 0] > 0)
                    & (prepared[test]["h1"][:, 1] >= 0)
                    & (prepared[test]["h1"][:, 2] >= 0)
                )
                scored = prepared[test]["scored"]
                output.append(
                    {
                        "test_index": test,
                        "fold": fold,
                        "horizon": horizon,
                        "train_indices": train,
                        "validation_index": validation,
                        "model": model,
                        "alpha": alpha,
                        "threshold": threshold,
                        "predicted_heads": prediction.tolist(),
                        "admitted_mass": mass.tolist(),
                        "fixed_threshold_curve": [
                            {
                                "threshold": threshold_value,
                                "coverage": float(
                                    gate_mass(
                                        prepared[test]["h1"],
                                        prediction,
                                        threshold_value,
                                    ).sum()
                                ),
                                "scored_utility": float(
                                    gate_mass(
                                        prepared[test]["h1"],
                                        prediction,
                                        threshold_value,
                                    )
                                    @ scored.mean(1)
                                ),
                            }
                            for threshold_value in DECLARATION["admission_thresholds"]
                        ],
                        "metrics": {
                            "minus_hold": float(mass @ scored.mean(1)),
                            "minus_forced": float((mass - greedy) @ scored.mean(1)),
                            "minus_replay": float(
                                (mass - greedy * replay) @ scored.mean(1)
                            ),
                            "retention": float(mass @ scored[:, 2]),
                            "coverage": float(mass.sum()),
                            "observed_beneficial_rejection_mass": float(
                                (greedy - mass) @ (scored.mean(1) > 0)
                            ),
                            "observed_harmful_admission_mass": float(
                                mass @ (scored.mean(1) < 0)
                            ),
                        },
                    }
                )
    return output


def stratified_indices(streams, draws=5000, seed=20261018, parents_per_stream=2):
    groups = [np.flatnonzero(np.asarray(streams) == s) for s in sorted(set(streams))]
    if len(groups) != 4 or any(len(g) != parents_per_stream for g in groups):
        raise ValueError("Declared balanced parents in four streams required")
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.choice(group, size=(draws, len(group))) for group in groups], axis=1
    )


def summarize(values, indices):
    values = np.asarray(values, float)
    if values.shape != (indices.shape[1],) or not np.isfinite(values).all():
        raise ValueError("Declared finite parent effects required")
    means = values[indices].mean(1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "units": len(values),
        "conditional_on_fixed_item_pool_and_fits": True,
    }


def planning(effects, strata=None):
    """Sensitivity only: pilot estimates are uncertain and CV fits overlap."""
    x = np.asarray(effects, float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("Two paired contrasts on at least eight parents required")
    covariance = np.cov(x, rowvar=False)
    if strata is not None:
        if len(strata) != len(x) or len(set(strata)) != 4:
            raise ValueError("Four matched stream strata required")
        residuals = np.zeros_like(x)
        for stream in sorted(set(strata)):
            indices = np.flatnonzero(np.asarray(strata) == stream)
            if len(indices) < 2:
                raise ValueError("Within-stream variance needs replication")
            residuals[indices] = x[indices] - x[indices].mean(0)
        covariance = residuals.T @ residuals / (len(x) - 4)
    sd = np.sqrt(np.maximum(0, np.diag(covariance)))
    if sd.max() < 1e-10:
        return {"status": "degenerate_pilot_variance_not_a_power_guarantee", "rows": []}
    z = NormalDist().inv_cdf(1 - 0.05 / 4)
    z90 = NormalDist().inv_cdf(0.9)
    rng = np.random.default_rng(20261018)
    noise = rng.multivariate_normal([0, 0], covariance, size=20000, check_valid="raise")
    rows = []
    for n in (32, 48, 64, 80, 96):
        for scale in (0.75, 1.0, 1.5, 2.0):
            for effect in (0.005, 0.01, 0.02):
                estimates = effect + noise * scale / np.sqrt(n)
                lower = estimates - z * sd * scale / np.sqrt(n)
                rows.append(
                    {
                        "parents": n,
                        "sd_multiplier": scale,
                        "assumed_effect_each_contrast_pp": effect * 100,
                        "joint_superiority_power_approx": float(
                            np.mean((lower > 0).all(1))
                        ),
                        "conservative_joint_80pct_MDE_pp": float(
                            (z + z90) * sd.max() * scale / np.sqrt(n) * 100
                        ),
                    }
                )
    return {
        "status": "normal_planning_only",
        "pilot_parents": len(x),
        "variance_basis": "within_fixed_stream"
        if strata is not None
        else "overall_parent",
        "rows": rows,
        "pilot_covariance": covariance.tolist(),
        "limitations": "No guarantee for new parents/items, fitted-policy uncertainty, retention noninferiority or practical point-estimate threshold. Small development cohorts and overlapping CV fits provide uncertain variance; draws never increase parent N.",
    }


def aggregate(states, gate_fits):
    streams = [s["stream"] for s in states]
    indices = stratified_indices(streams)
    metrics = {
        key: summarize([s["summary"]["metrics"][key] for s in states], indices)
        for key in states[0]["summary"]["metrics"]
    }
    forced = np.asarray([s["summary"]["metrics"]["forced_minus_hold"] for s in states])
    proxy = np.asarray([s["summary"]["metrics"]["proxy_minus_hold"] for s in states])
    better = proxy[indices].mean(1) - np.maximum(0, forced[indices].mean(1))
    proxy_summary = {
        "mean": float(proxy.mean() - max(0, forced.mean())),
        "ci95": np.quantile(better, [0.025, 0.975]).tolist(),
        "not_true_oracle_or_upper_bound": True,
    }
    interaction = summarize(
        [s["summary"]["interaction_agreement"] for s in states], indices
    )
    learned, projections = {}, {}
    for horizon in ("h1", "h2"):
        parent_metrics = []
        for i in range(8):
            rows = [
                r["metrics"]
                for r in gate_fits
                if r["test_index"] == i and r["horizon"] == horizon
            ]
            parent_metrics.append(
                {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            )
        learned[horizon] = {
            key: summarize([r[key] for r in parent_metrics], indices)
            for key in parent_metrics[0]
        }
        projections[horizon] = planning(
            [[r["minus_forced"], r["minus_hold"]] for r in parent_metrics]
        )
    per_stream = {}
    for stream in sorted(set(streams)):
        group = [s for s in states if s["stream"] == stream]
        per_stream[stream] = {
            "parents": len(group),
            "descriptive_only": True,
            "forced_minus_hold": float(
                np.mean([s["summary"]["metrics"]["forced_minus_hold"] for s in group])
            ),
            "proxy_minus_hold": float(
                np.mean([s["summary"]["metrics"]["proxy_minus_hold"] for s in group])
            ),
        }
    if proxy_summary["ci95"][1] < DECLARATION["practical_gain"]:
        decision = "STOP_BEFORE_C_no_demonstrated_operational_headroom"
    elif proxy_summary["mean"] >= DECLARATION["practical_gain"] and any(
        learned[h]["minus_forced"]["mean"] > 0 and learned[h]["minus_hold"]["mean"] > 0
        for h in learned
    ):
        decision = "REVIEW_B_development_signal_only_not_confirmation"
    else:
        decision = "REVIEW_ONLY_bounded_precision_option_no_confirmation_release"
    return {
        "declaration": DECLARATION,
        "metrics": metrics,
        "proxy_minus_better_constant": proxy_summary,
        "interaction_agreement": interaction,
        "pooling_sensitivity_eligible_development_only": interaction["ci95"][0] > 0,
        "learned_gates": learned,
        "power_planning": projections,
        "streams": per_stream,
        "decision": decision,
        "automatic_expansion": False,
        "independent_parents": 8,
        "development_only": True,
        "gate2_passed": None,
        "optimism_by_K": {
            str(k): summarize(
                [
                    s["summary"]["same_outcome_selection_optimism_by_K"][str(k)]
                    for s in states
                ],
                indices,
            )
            for k in (1, 2, 3)
        },
        "cross_draw_selection_gap_by_K": {
            str(k): summarize(
                [s["summary"]["cross_draw_selection_gap_by_K"][str(k)] for s in states],
                indices,
            )
            for k in (1, 2, 3)
        },
        "limitations": "Exploratory inspected eight-parent cohort. No intrinsic-unpredictability, calibrated-safety, universal-noise-floor, population causal or learned-superiority claim. Original source flags/labels stay unchanged; off-diagonal outcomes are diagnostic only.",
    }


def synthetic_controls(seed=20261018, trials=20000):
    """Known synthetic null/benefit; success does not establish real-data power."""
    rng = np.random.default_rng(seed)
    results = {}
    for name, truth in (
        ("equal_value_null", np.zeros((trials, 3))),
        (
            "recoverable_candidate_signal",
            np.broadcast_to([-0.06, 0.0, 0.06], (trials, 3)),
        ),
    ):
        selection = truth + rng.normal(0, 0.02, truth.shape)
        scoring = truth + rng.normal(0, 0.02, truth.shape)
        chosen = selection.argmax(1)
        apparent = scoring.max(1).mean()
        independent = scoring[np.arange(trials), chosen].mean()
        results[name] = {
            "same_outcome_maximum": float(apparent),
            "independent_selection_value": float(independent),
            "true_selected_value": float(truth[np.arange(trials), chosen].mean()),
        }
    truth = np.tile([-0.04, 0.04], trials // 2)
    selection = truth + rng.normal(0, 0.01, len(truth))
    scoring = truth + rng.normal(0, 0.01, len(truth))
    results["recoverable_admission"] = {
        "independent_admission_value": float(np.mean((selection > 0) * scoring)),
        "forced_value": float(scoring.mean()),
        "ideal_truth_value": float(np.maximum(truth, 0).mean()),
    }
    results["not_real_data_power"] = True
    return results


def conditional_item_bootstrap(states, gate_fits, draws=1000):
    """Shared item pools resample together; fitted policies stay frozen.

    This is scoring-item uncertainty conditional on decisions/checkpoints, not
    an independent-parent CI or a refit bootstrap. It must not be subtracted
    from parent variance as if the observed components were additive.
    """
    names = ("forced", "proxy", "h1_gate", "h2_gate")
    rng = np.random.default_rng(2026101801)
    shared, totals, points = {}, np.zeros((draws, 4)), np.zeros(4)
    for index, state in enumerate(states):
        assignment = folds(state["parent"])
        for fold in (0, 1):
            h1, future, _ = views(state["panel"], assignment, fold)
            greedy = choose(h1.mean(1))
            masses = []
            for horizon in ("h1", "h2"):
                matching = [
                    r
                    for r in gate_fits
                    if r["test_index"] == index
                    and r["fold"] == fold
                    and r["horizon"] == horizon
                ]
                if len(matching) != 1:
                    raise ValueError(
                        "One frozen gate decision per fold/parent required"
                    )
                masses.append(np.asarray(matching[0]["admitted_mass"]))
            for role in ROLES:
                indices = assignment[role][1 - fold]
                ids = tuple(
                    (
                        state["parent"]["items"][role][i]["id"],
                        state["parent"]["items"][role][i]["prompt_sha256"],
                    )
                    for i in indices
                )
                key = (role, ids)
                if key not in shared:
                    shared[key] = rng.integers(len(indices), size=(draws, len(indices)))
                role_panel = state["panel"][role]
                raw = (
                    role_panel["native"]
                    if "native" in role_panel
                    else role_panel["cube"][:, np.arange(3), np.arange(3)]
                )
                raw = raw[:, :, indices]
                mean = raw.mean(0)
                vectors = [greedy @ mean]
                admitted = []
                for held in range(3):
                    selection = future[[s for s in range(3) if s != held]].mean((0, 2))
                    admitted.append((greedy * (selection > 0)) @ raw[held])
                vectors.append(np.mean(admitted, axis=0))
                vectors.extend(mass @ mean for mass in masses)
                values = np.stack(vectors)
                weight = 1 / (len(states) * 2 * len(ROLES))
                totals += values[:, shared[key]].mean(2).T * weight
                points += values.mean(1) * weight
    contrasts = {name: (points[i], totals[:, i]) for i, name in enumerate(names)}
    for i, name in enumerate(names[1:], 1):
        contrasts[name + "_minus_forced"] = (
            points[i] - points[0],
            totals[:, i] - totals[:, 0],
        )
    return {
        "condition": "fixed_adapters_fitted_models_and_selection_decisions",
        "shared_item_pools_resampled_together": True,
        "distinct_role_half_pools": len(shared),
        "not_additive_to_parent_variance": True,
        "metrics": {
            name: {
                "mean": float(point),
                "bootstrap_sd": float(values.std(ddof=1)),
                "conditional_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            }
            for name, (point, values) in contrasts.items()
        },
    }
