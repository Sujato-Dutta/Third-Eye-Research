"""Cached selection value, observational mechanisms and known-signal controls."""

import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
import numpy as np
from third_eye.io import write_json
from third_eye.statistics.metrics import spearman
import diagnose_forecast_signal_v1 as analysis
from empirical_common_v1 import BASE, checked, snapshot_parts


def selection(rows):
    ix = analysis.groups(rows)
    one, two = analysis.targets(rows, "immediate"), analysis.targets(rows, "future")
    first, future = one @ analysis.WEIGHTS, two @ analysis.WEIGHTS
    values = {
        p: {
            k: []
            for k in [
                "future_utility",
                "oracle_regret",
                "harmful_utility",
                "target",
                "ood",
                "retention",
            ]
        }
        for p in ["immediate_oracle", "random", "future_oracle", "hold_parameters"]
    }
    advantage = []
    for g in ix:
        masks = {
            "immediate_oracle": np.isclose(
                first[g], np.max(first[g]), atol=1e-12, rtol=0
            ),
            "random": np.ones(3, bool),
            "future_oracle": np.isclose(
                future[g], np.max(future[g]), atol=1e-12, rtol=0
            ),
        }
        for policy, fields in values.items():
            if policy == "hold_parameters":
                consequence, utility, harm = np.zeros(3), 0.0, 0.0
            else:
                mask = masks[policy]
                consequence = two[g][mask].mean(0)
                utility = float(future[g][mask].mean())
                harm = float(np.mean(future[g][mask] < 0))
            fields["future_utility"].append(utility)
            fields["oracle_regret"].append(float(future[g].max() - utility))
            fields["harmful_utility"].append(harm)
            for name, score in zip(analysis.HEADS, consequence):
                fields[name].append(float(score))
        advantage.append(
            values["immediate_oracle"]["future_utility"][-1]
            - values["random"]["future_utility"][-1]
        )
    return {
        "policies": {
            p: {k: analysis.interval(v, rows) for k, v in fields.items()}
            for p, fields in values.items()
        },
        "immediate_oracle_minus_random": analysis.interval(advantage, rows),
        "note": "Observed outcomes; privileged oracles; hold skips both updates; regret relative to forced-update oracle may be negative for holding",
    }


def mechanism(rows):
    ix = analysis.groups(rows)
    one, two = analysis.targets(rows, "immediate"), analysis.targets(rows, "future")
    regret = []
    groups = {
        "probe_magnitude_dispersion": ("probe_adapter_delta_norm", "std"),
        "retention_interference": ("retention_gradient_cosine", "mean"),
        "data_diversity": ("embedding_diversity", "mean"),
        "candidate_loss_dispersion": ("pre_update_nll", "std"),
    }
    features = {k: [] for k in groups}
    for g in ix:
        u1, u2 = one[g] @ analysis.WEIGHTS, two[g] @ analysis.WEIGHTS
        winner = np.isclose(u1, u1.max(), atol=1e-12, rtol=0)
        regret.append(float(u2.max() - u2[winner].mean()))
        for name, (field, op) in groups.items():
            x = np.array(
                [rows[i]["precommit"]["features"].get(field, np.nan) for i in g]
            )
            features[name].append(float(x.std() if op == "std" else x.mean()))
    # Block bootstrap trajectory IDs; candidate rows are never independent units.
    trajectories = {}
    for j, g in enumerate(ix):
        trajectories.setdefault(rows[g[0]]["trajectory_id"], []).append(j)
    blocks = list(trajectories.values())
    result = {}
    for name, x in features.items():
        x, y = np.asarray(x), np.asarray(regret)
        valid = np.isfinite(x) & np.isfinite(y)
        rho = spearman(x[valid], y[valid])
        rng, boot = np.random.default_rng(42), []
        for _ in range(2000):
            j = np.concatenate(
                [blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]
            )
            j = j[valid[j]]
            if len(j) > 1:
                s = spearman(x[j], y[j])
                if s is not None:
                    boot.append(s)
        result[name] = {
            "spearman_with_observed_regret": rho,
            "trajectory_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist()
            if boot
            else None,
            "states": int(valid.sum()),
            "trajectories": len(blocks),
            "interpretation": "Exploratory association; not a causal mechanism or feature selection",
        }
    return result


def positive_controls(train, val):
    ix = analysis.groups(train)
    vx = analysis.groups(val)
    x = np.array([r["precommit"]["features"].get("pre_update_nll", 0) for r in train])
    z = np.array([r["precommit"]["features"].get("pre_update_nll", 0) for r in val])
    scale = max(float(x.std()), 1e-6)
    x, z = (x - x.mean()) / scale, (z - x.mean()) / scale
    # Center separately by candidate set; coefficients fixed before outcomes.
    x[ix] -= x[ix].mean(1, keepdims=True)
    z[vx] -= z[vx].mean(1, keepdims=True)
    result = []
    for noise in [0.0, 0.5, 1.0]:
        for seed in [42, 43, 44]:
            rng = np.random.default_rng(seed)
            a, b = copy.deepcopy(train), copy.deepcopy(val)
            for rows, s in [(a, x), (b, z)]:
                synthetic = 0.01 * (s + noise * rng.normal(size=len(s)))
                for i, r in enumerate(rows):
                    r["labels"]["h2"] = {h: float(synthetic[i]) for h in analysis.HEADS}
            pred = analysis.ridge(a, b, "future", True)
            metrics = analysis.scores(b, analysis.targets(b, "future"), pred)
            result.append(
                {
                    "seed": seed,
                    "injected_noise_sd": noise,
                    "metrics": metrics["utility"],
                    "synthetic": True,
                    "real_forecasting_success": False,
                }
            )
    return result


def main():
    checked()
    _, parts = snapshot_parts()
    output = BASE / "cached_analysis"
    if output.exists():
        raise FileExistsError("Preserve existing cached analysis")
    output.mkdir()
    selection_reports = {p: selection(rows) for p, rows in parts.items()}
    write_json(output / "selection_value.json", selection_reports)
    write_json(
        output / "mechanism_associations.json",
        {p: mechanism(rows) for p, rows in parts.items() if p != "test"},
    )
    write_json(
        output / "synthetic_positive_controls.json",
        positive_controls(parts["train"], parts["validation"]),
    )
    old = json.loads((ROOT / "runs/a2/signal_confirmation_v1/review.json").read_text())
    spread = old["paired_comparisons"]["immediate_own_minus_future_own"]
    write_json(
        output / "detectability.json",
        {
            "independent_units": spread["units"],
            "illustrative_normal_approx_80pct_detectable_difference": (1.96 + 0.8416)
            * spread["std"]
            / math.sqrt(spread["units"]),
            "limitations": "Normal approximation on eight trajectory differences; not proof of real negative-result power",
            "multiple_tests": "Mechanism associations descriptive, no uncorrected primary significance claims",
        },
    )
    write_json(
        output / "completed.json",
        {
            "status": "cached_empirical_review_required",
            "new_gpu_jobs": 0,
            "old_test_exposed": True,
            "new_forecaster_selection": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "cached_empirical_review_required",
                "heldout_selection": selection_reports["test"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
