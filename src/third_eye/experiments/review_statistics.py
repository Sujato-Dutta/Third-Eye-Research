"""Review-driven paired inference; preserve parent and shared-item dependence."""

from pathlib import Path

import numpy as np

from third_eye.io import write_json
from third_eye.statistics.inference import holm

ROLES = ("target_test", "ood_test", "retention_test")


def bootstrap_heads(states, matrices, draws=5000):
    if len(states) != len(matrices) or not states:
        raise ValueError("Complete paired state matrices required")
    groups = sorted({s["stream"] for s in states})
    sizes = [sum(s["stream"] == g for s in states) for g in groups]
    if len(groups) != 4 or len(set(sizes)) != 1:
        raise ValueError("Four balanced original model/task streams required")
    rng = np.random.default_rng(20261026)
    weights = np.zeros((draws, len(states)))
    for group in groups:
        indices = np.array([i for i, s in enumerate(states) if s["stream"] == group])
        selected = rng.choice(indices, (draws, len(indices)))
        for i in indices:
            weights[:, i] = (selected == i).sum(1) / len(states)
    endpoints = matrices[0][ROLES[0]].shape[0]
    point, boot = np.zeros((3, endpoints)), np.zeros((draws, 3, endpoints))
    subgroup_point = {g: np.zeros((3, endpoints)) for g in groups}
    subgroup_boot = {g: np.zeros((draws, 3, endpoints)) for g in groups}
    shared, pools = {}, {}
    for i, (state, matrix) in enumerate(zip(states, matrices)):
        for r, role in enumerate(ROLES):
            identity = tuple(tuple(v) for v in state["ids"][role])
            if any(
                set(other) & set(identity) and other != identity
                for other in pools.get(role, [])
            ):
                raise ValueError(
                    "Partially shared item pools need explicit identity resampling"
                )
            pools.setdefault(role, []).append(identity)
            x = np.asarray(matrix[role], float)
            if x.shape != (endpoints, len(identity)) or not np.isfinite(x).all():
                raise ValueError("Finite complete item matrices required")
            key = (role, identity)
            if key not in shared:
                shared[key] = rng.integers(len(identity), size=(draws, len(identity)))
            sample = x[:, shared[key]].mean(2).T
            point[r] += x.mean(1) / len(states)
            boot[:, r] += sample * weights[:, i, None]
            subgroup_point[state["stream"]][r] += x.mean(1) / sizes[0]
            subgroup_boot[state["stream"]][:, r] += sample * weights[:, i, None] * 4
    return point, boot, subgroup_point, subgroup_boot


def summarize(states, matrices, names, output, metadata, adjusted=False):
    point, boot, subpoint, subboot = bootstrap_heads(states, matrices)
    avg, samples = point.mean(0), boot.mean(1)

    def interval(p, b):
        return {
            "mean_pp": float(100 * p),
            "ci95_pp": (100 * np.quantile(b, [0.025, 0.975])).tolist(),
        }

    result = {
        **metadata,
        "parents": len(states),
        "parents_per_stream": len(states) // 4,
        "units": "percentage_points",
        "inference": "balanced parent resampling and shared exact item identity resampling; draws nested within parent",
        "aggregate": {
            name: interval(avg[i], samples[:, i]) for i, name in enumerate(names)
        },
        "heads": {
            role: {
                name: interval(point[r, i], boot[:, r, i])
                for i, name in enumerate(names)
            }
            for r, role in enumerate(ROLES)
        },
        "streams_descriptive": {
            g: {
                name: interval(p.mean(0)[i], subboot[g].mean(1)[:, i])
                for i, name in enumerate(names)
            }
            for g, p in subpoint.items()
        },
        "original_primary_unchanged": True,
        "forecaster_fits": 0,
    }
    if adjusted:
        pvalues = {
            name: float(
                (1 + np.sum(np.abs(samples[:, i] - avg[i]) >= abs(avg[i]) - 1e-12))
                / (len(samples) + 1)
            )
            for i, name in enumerate(names)
        }
        corrected = holm(pvalues)
        for i, name in enumerate(names):
            result["aggregate"][name].update(
                holm_p=corrected[name],
                simultaneous_ci95_pp=(
                    100
                    * np.quantile(
                        samples[:, i],
                        [0.05 / (2 * len(names)), 1 - 0.05 / (2 * len(names))],
                    )
                ).tolist(),
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    lines = [
        "# Review-driven paired evidence",
        "",
        f"{len(states)} parents; original confirmation results remain separate.",
        "",
        "| Contrast | Mean (pp) | Joint95% interval (pp) |",
        "| --- | ---: | --- |",
    ]
    for name, row in result["aggregate"].items():
        lines.append(
            f"| {name} | {row['mean_pp']:.3f} | [{row['ci95_pp'][0]:.3f}, {row['ci95_pp'][1]:.3f}] |"
        )
    (output / "findings.md").write_text("\n".join(lines) + "\n")
    return result


def control_matrix(cells, draws, role):
    x = {name: np.asarray(values[role], float) for name, values in cells.items()}
    native = np.mean([x[f"draw{d}_m1_d1_50"] for d in draws], axis=0)
    first = np.mean([x[f"draw{d}_m0_d0_50"] for d in draws], axis=0)
    m0d1 = np.mean([x[f"draw{d}_m0_d1_50"] for d in draws], axis=0)
    m1d0 = np.mean([x[f"draw{d}_m1_d0_50"] for d in draws], axis=0)
    recipient_d1 = native - x["m1_before"] - m0d1 + x["m0_before"]
    recipient_d0 = m1d0 - x["m1_before"] - first + x["m0_before"]
    return np.stack(
        [
            native - x["m1_before"],
            first - x["m0_before"],
            recipient_d1,
            recipient_d0,
            recipient_d1 - recipient_d0,
            np.mean([x[f"draw{d}_m1_d1_10"] for d in draws], axis=0) - native,
            np.mean([x[f"draw{d}_m1_d1_25"] for d in draws], axis=0) - native,
            np.mean([x[f"draw{d}_m1_gold_50"] for d in draws], axis=0) - native,
            x["m1_pooled_50"] - native,
        ]
    )


CONTROL_NAMES = (
    "native_continuation_increment",
    "native_first_update_increment",
    "recipient_increment_difference_D1",
    "recipient_increment_difference_D0",
    "recipient_by_donor_interaction",
    "steps10_minus_steps50",
    "steps25_minus_steps50",
    "gold_minus_self_generated",
    "pooled_minus_size2",
)
