"""Conditional precision/cost planning before any fresh confirmation outcomes."""

from statistics import NormalDist

import numpy as np

from third_eye.experiments.mechanism import ROLES, folds


DECLARATION = {
    "version": "confirmation_design_20261022",
    "development_only": True,
    "automatic_expansion": False,
    "orientation": 0,
    "orientation_reason": "fixed first hash half for selection; never choose an orientation by results",
    "policies": [
        "forced",
        "replay",
        "ridge_h1",
        "ridge_h2",
        "backbone_h1",
        "backbone_h2",
    ],
    "parents": [48, 64, 32],
    "target_ood_sizes": [128, 256, 384],
    "retention_size": 256,
    "bootstrap_draws": 2000,
    "practical_effect": 0.005,
    "required_planning_power": 0.8,
    "measurement_primary": [
        "forced_future_utility_below_hold",
        "qwen_minus_llama_forced_future_utility",
    ],
    "effect_shrinkage": [0.5, 0.75, 1.0],
    "sd_inflation": [1.0, 1.25, 1.5],
    "design_gate_shrinkage": 0.75,
    "design_gate_sd_inflation": 1.25,
    "simultaneous_critical": "normal_1_minus_0.05_over_4",
    "confirmation_ceiling_if_recursion_omitted_sus": 810,
    "reserved_cost_calibration_humaneval_items": 8,
    "unmeasured_humaneval_cost_multiplier": 4,
    "final_checkpoints_per_parent": 13,
    "cost_margin": 1.25,
    "limitations": "Empirical fixed-action parent/item planning, not real-data power or new-domain validation. Development OOD is a proxy for code final OOD. No source data or method is confirmed by projections.",
}


def policy_actions(records, count=16):
    names = DECLARATION["policies"]
    result = np.zeros((count, len(names), 3))
    # Forced/replay are reconstructed by the caller from selection items.
    for index in range(count):
        for offset, name in enumerate(names[2:], 2):
            rows = [
                r
                for r in records
                if r["policy"] == name
                and r["test_index"] == index
                and r["fold"] == 0
                and r["scheme"] == "leave_parent"
            ]
            if len(rows) != 1:
                raise ValueError("One source-bound decision per parent/policy required")
            mass = np.asarray(rows[0]["admitted_mass"], float)
            if (
                mass.shape != (3,)
                or not np.isfinite(mass).all()
                or (mass < 0).any()
                or mass.sum() > 1 + 1e-9
            ):
                raise ValueError("Finite sub-probability admission actions required")
            result[index, offset] = mass
    return result


def bootstrap(states, actions, parents, item_counts, draws=2000):
    """Use only disjoint scoring-half items; shared identities resample together."""
    if len(states) != 16 or parents % 4 or actions.shape != (16, 6, 3):
        raise ValueError("Balanced frozen sixteen-parent action panel required")
    rng = np.random.default_rng(20261022 + parents + sum(item_counts.values()))
    counts = np.zeros((draws, 16))
    streams = [state["stream"] for state in states]
    for stream in sorted(set(streams)):
        group = np.flatnonzero(np.asarray(streams) == stream)
        if len(group) != 4:
            raise ValueError("Four development parents per stream required")
        selected = rng.choice(group, size=(draws, parents // 4))
        for index in group:
            counts[:, index] = (selected == index).sum(1) / parents
    shared = {}
    point, boot = np.zeros(7), np.zeros((draws, 7))
    rpoint, rboot = np.zeros(6), np.zeros((draws, 6))
    for index, state in enumerate(states):
        task = state["stream"].split("/")[1]
        sign = 2 if state["stream"].split("/")[0].startswith("qwen") else -2
        assignment = folds(state["parent"])
        for role in ROLES:
            selected = assignment[role][1]
            identities = tuple(
                (
                    state["parent"]["items"][role][i]["id"],
                    state["parent"]["items"][role][i]["prompt_sha256"],
                )
                for i in selected
            )
            target_count = item_counts[task + "/" + role]
            key = (role, identities, target_count)
            if key not in shared:
                shared[key] = rng.integers(len(selected), size=(draws, target_count))
            raw = state["panel"][role]["native"].mean(0)[:, selected]
            values = actions[index] @ raw
            augmented = np.concatenate([values, sign * values[:1]], axis=0)
            point += augmented.mean(1) / (16 * 3)
            sampled = augmented[:, shared[key]].mean(2).T
            boot += sampled * counts[:, index, None] / 3
            if role == "retention_dev":
                rpoint += values.mean(1) / 16
                rboot += sampled[:, :6] * counts[:, index, None]
    return point, boot, rpoint, rboot


def conditional_power(point, boot, shrinkage, inflation):
    critical = NormalDist().inv_cdf(1 - 0.05 / 4)
    contrasts = np.column_stack((-boot[:, 0], boot[:, 6]))
    center = np.array([-point[0], point[6]])
    noise = (contrasts - center) * inflation
    sd = contrasts.std(0, ddof=1) * inflation
    measurement = float(
        np.mean((center * shrinkage + noise - critical * sd > 0).all(1))
    )
    methods = {}
    better_constant = max(0.0, point[0])
    target = np.array([better_constant + 0.005, better_constant + 0.005 - point[0]])
    for offset, name in enumerate(DECLARATION["policies"][2:], 2):
        pair = np.column_stack((boot[:, offset], boot[:, offset] - boot[:, 0]))
        mean = np.array([point[offset], point[offset] - point[0]])
        residual = (pair - mean) * inflation
        se = pair.std(0, ddof=1) * inflation
        methods[name] = {
            "joint_power_at_practical_effect": float(
                np.mean((target + residual - critical * se > 0).all(1))
            ),
            "contrast_sd_pp": (se * 100).tolist(),
        }
    return {
        "measurement_joint_power": measurement,
        "measurement_effects_pp": (center * shrinkage * 100).tolist(),
        "measurement_sd_pp": (sd * 100).tolist(),
        "admission_counterfactual_effect_pp": (target * 100).tolist(),
        "admission_planning": methods,
    }


def price(parents, item_counts, base_limits, benchmarks, humaneval_rates=None):
    if len(benchmarks) != 4 or {b["task"]["stream"] for b in benchmarks} != set(
        base_limits
    ):
        raise ValueError("All four measured stream benchmarks required")
    if parents % 4 or any(type(v) is not int or v <= 0 for v in item_counts.values()):
        raise ValueError("Balanced N and positive integer item counts required")
    limits = {}
    for benchmark in benchmarks:
        stream = benchmark["task"]["stream"]
        task = stream.split("/")[1]
        rates = {}
        for row in benchmark["observations"]:
            rates[row["role"]] = max(
                rates.get(row["role"], 0),
                (row["generation_seconds"] + row["verification_seconds"])
                / row["count"],
            )
        if task == "code":
            rates["ood_dev"] = (
                humaneval_rates[stream] if humaneval_rates else rates["ood_dev"] * 4
            )
        evaluation = sum(rates[role] * item_counts[task + "/" + role] for role in ROLES)
        restore = max(row["restore_seconds"] for row in benchmark["observations"])
        extra = 1.25 * (
            13 * (evaluation + restore) + benchmark["model_load_seconds"] + 120
        )
        limits[stream] = int(np.ceil((base_limits[stream] + extra) / 900) * 900)
    return {
        "walltime_seconds_by_stream": limits,
        "base_plus_scoring_reservation_sus": (parents // 4)
        * sum(limits.values())
        / 3600,
        "includes": "thirteen final-scored checkpoints, measured generation/verifier/restore/load rates, margin and overhead, plus full conservative base collection",
        "humaneval_directly_measured": humaneval_rates is not None,
    }


def choose_design(rows, method_promising, remaining_sus):
    """Fixed preference order; never call a measurement result method efficacy."""
    for parents in DECLARATION["parents"]:
        candidates = [
            row
            for row in rows
            if row["parents"] == parents
            and row["cost"]["base_plus_scoring_reservation_sus"] + 2
            <= min(810, remaining_sus - 100)
        ]
        for row in sorted(
            candidates,
            key=lambda value: value["cost"]["base_plus_scoring_reservation_sus"],
        ):
            gate = next(
                v
                for v in row["sensitivity"]
                if v["effect_shrinkage"] == 0.75 and v["sd_inflation"] == 1.25
            )
            if (
                method_promising
                and gate["admission_planning"]["ridge_h1"][
                    "joint_power_at_practical_effect"
                ]
                >= 0.8
            ):
                return {
                    "purpose": "admission_confirmation_design_candidate",
                    "design": row,
                }
            if gate["measurement_joint_power"] >= 0.8:
                return {
                    "purpose": "measurement_confirmation_design_candidate_not_method_efficacy",
                    "design": row,
                }
    return None
