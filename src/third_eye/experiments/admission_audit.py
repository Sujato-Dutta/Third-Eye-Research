"""Exploratory saved-data admission audit; no new architecture or confirmation."""

from statistics import NormalDist

import numpy as np

from third_eye.experiments.admission import (
    DECLARATION,
    gate_mass,
    ridge_fit,
    ridge_predict,
    stratified_indices,
    summarize,
    views,
)
from third_eye.experiments.mechanism import ROLES, choose, folds


DECLARED = {
    "version": "admission_decision_audit_20261021",
    "development_only": True,
    "previously_inspected_outcomes": True,
    "automatic_expansion": False,
    "families": ["global", "backbone", "task", "stream", "ridge", "stream_residual"],
    "horizons": ["h1", "h2"],
    "cv": ["leave_parent", "leave_seed"],
    "context_baselines": "explicit categories only; not evidence of context-free prediction",
    "features": ["h1_target", "h1_ood", "h1_retention", "log1p_effective_update_norm"],
    "ridge_alphas": DECLARATION["ridge_alphas"],
    "thresholds": DECLARATION["admission_thresholds"],
    "constant_threshold": 0,
    "retention_floor": -0.005,
    "item_folds": "unchanged hash halves; all fit/validation targets use selection half only",
    "parent_bootstrap": 5000,
    "crossed_bootstrap": 1000,
    "planning_N": [16, 48, 64, 96],
    "hypothetical_item_multipliers": [1, 2, 4],
    "assumed_effects": [0.005, 0.01, 0.02],
    "uncertainty": "resample shared item pools together and parents within fixed streams; fixed fits/actions, not additive variance components",
    "confirmation": "no release from this exploratory report; requires independent policy, precision, item-exposure and cost review",
}


def prepare(states, fold):
    if len(states) != 16 or len({s["trajectory_id"] for s in states}) != 16:
        raise ValueError("Sixteen distinct development parents required")
    output = []
    for state in states:
        assignment = folds(state["parent"])
        h1, future, _ = views(state["panel"], assignment, fold)
        _, scored, _ = views(state["panel"], assignment, 1 - fold)
        norm = np.asarray(state["effective_update_norms"], float)
        if norm.shape != (3,) or not np.isfinite(norm).all() or (norm < 0).any():
            raise ValueError("Actual finite effective norms required")
        backbone, task = state["stream"].split("/")
        output.append(
            {
                "x": np.column_stack((h1, np.log1p(norm))),
                "h1": h1,
                "h2": future.mean(0),
                "scored": scored.mean(0),
                "stream": state["stream"],
                "backbone": backbone,
                "task": task,
                "global": "all",
                "seed": state["seed"],
            }
        )
    return output


def splits(prepared, scheme):
    if scheme == "leave_parent":
        return [
            (
                [i for i in range(16) if i not in (test, (test + 4) % 16)],
                [(test + 4) % 16],
                [test],
            )
            for test in range(16)
        ]
    if scheme != "leave_seed":
        raise ValueError("Unknown declared split")
    seeds = sorted({p["seed"] for p in prepared})
    if len(seeds) != 4 or any(
        sum(p["seed"] == s for p in prepared) != 4 for s in seeds
    ):
        raise ValueError("Four shared seed groups of four parents required")
    result = []
    for index, seed in enumerate(seeds):
        validation_seed = seeds[(index + 1) % 4]
        result.append(
            (
                [
                    i
                    for i, p in enumerate(prepared)
                    if p["seed"] not in (seed, validation_seed)
                ],
                [i for i, p in enumerate(prepared) if p["seed"] == validation_seed],
                [i for i, p in enumerate(prepared) if p["seed"] == seed],
            )
        )
    return result


def selected_mean(prepared, indices, horizon):
    return np.mean(
        [choose(prepared[i]["h1"].mean(1)) @ prepared[i][horizon] for i in indices],
        axis=0,
    )


def fit(prepared, indices, horizon, family, alpha=1):
    group = "stream" if family == "stream_residual" else family
    if family == "ridge":
        return {
            "family": family,
            "ridge": ridge_fit(
                np.concatenate([prepared[i]["x"] for i in indices]),
                np.concatenate([prepared[i][horizon] for i in indices]),
                alpha,
            ),
        }
    groups = sorted({prepared[i][group] for i in indices})
    baselines, centers = {}, {}
    for key in groups:
        selected = [i for i in indices if prepared[i][group] == key]
        baselines[key] = selected_mean(prepared, selected, horizon)
        centers[key] = np.concatenate([prepared[i]["x"] for i in selected]).mean(0)
    model = {
        "family": family,
        "group": group,
        "baselines": baselines,
        "centers": centers,
        "fallback": selected_mean(prepared, indices, horizon),
        "fallback_center": np.concatenate([prepared[i]["x"] for i in indices]).mean(0),
    }
    if family == "stream_residual":
        model["ridge"] = ridge_fit(
            np.concatenate(
                [prepared[i]["x"] - centers[prepared[i][group]] for i in indices]
            ),
            np.concatenate(
                [prepared[i][horizon] - baselines[prepared[i][group]] for i in indices]
            ),
            alpha,
        )
    return model


def predict(model, parent):
    if model["family"] == "ridge":
        return ridge_predict(model["ridge"], parent["x"])
    key = parent[model["group"]]
    baseline = model["baselines"].get(key, model["fallback"])
    result = np.broadcast_to(baseline, (3, 3)).copy()
    if model["family"] == "stream_residual":
        center = model["centers"].get(key, model["fallback_center"])
        result += ridge_predict(model["ridge"], parent["x"] - center)
    return result


def correlation(a, b):
    def ranks(values):
        values = np.round(values, 12)
        return np.array([sum(values < v) + (sum(values == v) - 1) / 2 for v in values])

    x, y = ranks(a), ranks(b)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(states, scheme):
    """Both label horizons use the same four inputs; context rules are baselines."""
    records = []
    for fold in (0, 1):
        prepared = prepare(states, fold)
        for train, validation, tests in splits(prepared, scheme):
            train_x = np.concatenate([prepared[i]["x"] for i in train])
            for family in DECLARED["families"]:
                for horizon in DECLARED["horizons"]:
                    alphas = (
                        DECLARED["ridge_alphas"]
                        if family in ("ridge", "stream_residual")
                        else [1.0]
                    )
                    candidates = []
                    for alpha in alphas:
                        model = fit(prepared, train, horizon, family, alpha)
                        error = np.mean(
                            [
                                np.mean(
                                    (predict(model, prepared[i]) - prepared[i][horizon])
                                    ** 2
                                )
                                for i in validation
                            ]
                        )
                        candidates.append((float(error), -alpha, model))
                    _, negative_alpha, model = min(
                        candidates, key=lambda v: (v[0], v[1])
                    )
                    thresholds = (
                        DECLARED["thresholds"]
                        if family in ("ridge", "stream_residual")
                        else [0.0]
                    )
                    values = []
                    for threshold in thresholds:
                        value = np.mean(
                            [
                                gate_mass(
                                    prepared[i]["h1"],
                                    predict(model, prepared[i]),
                                    threshold,
                                )
                                @ prepared[i][horizon].mean(1)
                                for i in validation
                            ]
                        )
                        values.append((round(float(value), 12), threshold))
                    _, threshold = max(values)
                    for index in tests:
                        parent = prepared[index]
                        prediction = predict(model, parent)
                        mass = gate_mass(parent["h1"], prediction, threshold)
                        greedy = choose(parent["h1"].mean(1))
                        score = parent["scored"].mean(1)
                        x = parent["x"]
                        outside = ((x < train_x.min(0)) | (x > train_x.max(0))).mean(0)
                        records.append(
                            {
                                "policy": family + "_" + horizon,
                                "scheme": scheme,
                                "fold": fold,
                                "test_index": index,
                                "train_indices": train,
                                "validation_indices": validation,
                                "alpha": -negative_alpha,
                                "threshold": threshold,
                                "prediction": prediction.tolist(),
                                "admitted_mass": mass.tolist(),
                                "feature_outside_train_range": outside.tolist(),
                                "within_parent_rank_rho": correlation(
                                    prediction.mean(1), score
                                ),
                                "candidate_mean_squared_error": float(
                                    np.mean((prediction.mean(1) - score) ** 2)
                                ),
                                "metrics": {
                                    "minus_hold": float(mass @ score),
                                    "minus_forced": float((mass - greedy) @ score),
                                    "retention": float(mass @ parent["scored"][:, 2]),
                                    "coverage": float(mass.sum()),
                                    "harmful_admission_mass": float(mass @ (score < 0)),
                                    "beneficial_rejection_mass": float(
                                        (greedy - mass) @ (score > 0)
                                    ),
                                },
                            }
                        )
    return records


def summarize_records(states, records):
    indices = stratified_indices([s["stream"] for s in states], parents_per_stream=4)
    output = {}
    for name in sorted({r["policy"] for r in records}):
        selected = [r for r in records if r["policy"] == name]
        values = []
        for index in range(16):
            parent = [r for r in selected if r["test_index"] == index]
            if len(parent) != 2:
                raise ValueError(
                    "Exactly two orientation decisions per parent required"
                )
            values.append(
                {
                    key: float(np.mean([r["metrics"][key] for r in parent]))
                    for key in parent[0]["metrics"]
                }
            )
        valid = [
            r["within_parent_rank_rho"]
            for r in selected
            if r["within_parent_rank_rho"] is not None
        ]
        output[name] = {
            "metrics": {
                key: summarize([v[key] for v in values], indices) for key in values[0]
            },
            "parent_effects": values,
            "rank_rho_descriptive": float(np.mean(valid)) if valid else None,
            "valid_rank_orientations": len(valid),
            "candidate_mean_squared_error": float(
                np.mean([r["candidate_mean_squared_error"] for r in selected])
            ),
            "feature_extrapolation_fraction": np.mean(
                [r["feature_outside_train_range"] for r in selected], axis=0
            ).tolist(),
        }
    for horizon in ("h1", "h2"):
        difference = [
            a["minus_hold"] - b["minus_hold"]
            for a, b in zip(
                output["stream_residual_" + horizon]["parent_effects"],
                output["stream_" + horizon]["parent_effects"],
            )
        ]
        output["stream_residual_" + horizon]["minus_stream_constant"] = summarize(
            difference, indices
        )
    return output


def reproduce(original, records):
    for row in original:
        matches = [
            r
            for r in records
            if r["scheme"] == "leave_parent"
            and r["policy"] == "ridge_" + row["horizon"]
            and r["test_index"] == row["test_index"]
            and r["fold"] == row["fold"]
        ]
        if len(matches) != 1:
            raise ValueError("Original gate fit matching failed")
        new = matches[0]
        if (
            new["alpha"] != row["alpha"]
            or new["threshold"] != row["threshold"]
            or not np.allclose(
                new["prediction"], row["predicted_heads"], atol=1e-9, rtol=1e-9
            )
            or not np.array_equal(new["admitted_mass"], row["admitted_mass"])
        ):
            raise ValueError("Audit failed to reproduce original B gate decisions")
    return True


def action_tensor(states, records):
    names = ["hold", "forced", "replay"] + sorted({r["policy"] for r in records})
    actions = np.zeros((len(names), 16, 2, 3))
    for fold in (0, 1):
        prepared = prepare(states, fold)
        for index, parent in enumerate(prepared):
            greedy = choose(parent["h1"].mean(1))
            actions[1, index, fold] = greedy
            actions[2, index, fold] = greedy * (
                (parent["h1"][:, 0] > 0)
                & (parent["h1"][:, 1] >= 0)
                & (parent["h1"][:, 2] >= 0)
            )
    for row in records:
        actions[names.index(row["policy"]), row["test_index"], row["fold"]] = row[
            "admitted_mass"
        ]
    return names, actions


def crossed_bootstrap(
    states, actions, parents=16, item_multiplier=1, mode="joint", draws=1000
):
    """Conditional crossed resampling; hypothetical larger suites assume IID items.

    Shared items are drawn once per distinct role/half pool for every parent.
    Existing folds, candidate selection, fitted policies and adapters stay fixed.
    This does not produce additional independent observations or refit uncertainty.
    """
    if parents % 4 or mode not in ("parent", "item", "joint"):
        raise ValueError("Balanced parent count and declared resampling mode required")
    rng = np.random.default_rng(20261021 + parents + int(item_multiplier) * 100)
    counts = np.ones((draws, 16)) / 16
    if mode != "item":
        streams = [s["stream"] for s in states]
        counts = np.zeros((draws, 16))
        for stream in sorted(set(streams)):
            group = np.flatnonzero(np.asarray(streams) == stream)
            selected = rng.choice(group, size=(draws, parents // 4))
            for i in group:
                counts[:, i] = (selected == i).sum(1) / parents
    shared, boot, retention = (
        {},
        np.zeros((draws, len(actions))),
        np.zeros((draws, len(actions))),
    )
    point, retention_point = np.zeros(len(actions)), np.zeros(len(actions))
    for index, state in enumerate(states):
        assignment = folds(state["parent"])
        for fold in (0, 1):
            for role in ROLES:
                selected = assignment[role][1 - fold]
                ids = tuple(
                    (
                        state["parent"]["items"][role][i]["id"],
                        state["parent"]["items"][role][i]["prompt_sha256"],
                    )
                    for i in selected
                )
                key = (role, ids)
                raw = state["panel"][role]["native"].mean(0)[:, selected]
                values = actions[:, index, fold] @ raw
                point += values.mean(1) / (16 * 2 * 3)
                if mode == "parent":
                    sampled = np.broadcast_to(values.mean(1), (draws, len(actions)))
                else:
                    if key not in shared:
                        shared[key] = rng.integers(
                            len(selected),
                            size=(
                                draws,
                                max(1, round(len(selected) * item_multiplier)),
                            ),
                        )
                    sampled = values[:, shared[key]].mean(2).T
                boot += sampled * counts[:, index, None] / 6
                if role == "retention_dev":
                    retention += sampled * counts[:, index, None] / 2
                    retention_point += values.mean(1) / (16 * 2)
    return point, boot, retention_point, retention


def uncertainty(states, records):
    names, actions = action_tensor(states, records)
    results = {}
    z = NormalDist().inv_cdf(1 - 0.05 / 4)
    for mode in ("parent", "item", "joint"):
        point, boot, rpoint, rboot = crossed_bootstrap(states, actions, mode=mode)
        results[mode] = {
            name: {
                "mean": float(point[i]),
                "conditional_ci95": np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
                "conditional_sd": float(boot[:, i].std(ddof=1)),
                "retention_mean": float(rpoint[i]),
                "retention_conditional_ci95": np.quantile(
                    rboot[:, i], [0.025, 0.975]
                ).tolist(),
            }
            for i, name in enumerate(names)
        }
    planning = []
    for n in DECLARED["planning_N"]:
        for factor in DECLARED["hypothetical_item_multipliers"]:
            point, boot, _, _ = crossed_bootstrap(
                states, actions, parents=n, item_multiplier=factor
            )
            for i, name in enumerate(names[3:], 3):
                if not 0 < actions[i].sum(-1).mean() < 1:
                    continue
                contrasts = np.column_stack((boot[:, i] - boot[:, 1], boot[:, i]))
                center = np.array([point[i] - point[1], point[i]])
                noise, sd = contrasts - center, contrasts.std(0, ddof=1)
                if sd.max() < 1e-12:
                    continue
                for effect in DECLARED["assumed_effects"]:
                    power = np.mean(((effect + noise - z * sd) > 0).all(1))
                    planning.append(
                        {
                            "policy": name,
                            "parents": n,
                            "hypothetical_item_multiplier": factor,
                            "assumed_effect_each_contrast_pp": effect * 100,
                            "joint_superiority_power_approx": float(power),
                            "conditional_sd_pp": (sd * 100).tolist(),
                            "limitations": "Fixed-action empirical-parent/item sensitivity, no policy refitting or new-item/domain uncertainty; hypothetical extra items are not real data. Retention and practical point threshold are separate.",
                        }
                    )
    return {
        "conditional_uncertainty": results,
        "power_sensitivity": planning,
        "limitations": "Parents and shared items crossed directly; do not add/subtract the conditional variances. No simultaneous confirmation or automatic release from these exploratory projections.",
    }
