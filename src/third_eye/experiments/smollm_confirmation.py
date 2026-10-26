"""Fixed third-family extension: measured horizons, with no learned policy."""

from pathlib import Path

import numpy as np

from third_eye.experiments.mechanism import choose
from third_eye.io import write_json

DECLARATION = {
    "version": "smollm16_20261024",
    "model": "HuggingFaceTB/SmolLM3-3B",
    "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
    "seeds": {
        "code": list(range(2026102900, 2026102908)),
        "math": list(range(2026102800, 2026102808)),
    },
    "parents": 16,
    "parents_per_task": 8,
    "continuation_draws": 3,
    "scoring_items": {"target_test": 128, "ood_test": 128, "retention_test": 256},
    "selection_orientation": 0,
    "bootstrap_draws": 5000,
    "primary": "selected_future_minus_selected_immediate_utility",
    "primary_test": "two_sided_95pct_joint_run_shared_item_bootstrap_interval",
    "selection": "development_H1_equal_head_utility_uniform_ties",
    "terminal": "M2_equals_M1_no_fabricated_training",
    "forecaster_fits": 0,
    "automatic_expansion": False,
    "excluded_future_models": ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"],
    "future_models_execution": "DGX_when_access_returns_separate_frozen_extension",
    "additional_revision_su_cap": 500,
    "parent_array_su_cap": 224,
    "per_parent_seconds": 50400,
}


def actions(stream, immediate):
    x = np.asarray(immediate, float)
    if (
        not stream.startswith("smollm3-3b/")
        or x.shape != (3, 3)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Three finite SmolLM development candidates required")
    mass = choose(x.mean(1))
    return {
        "hold": [0.0] * 3,
        "forced": mass.tolist(),
        "replay": (mass * ((x[:, 0] > 0) & (x[:, 1] >= 0) & (x[:, 2] >= 0))).tolist(),
    }


NAMES = (
    "selected_h1",
    "selected_h2",
    "uniform_h1",
    "uniform_h2",
    "selected_change",
    "uniform_change",
    "replay_h2",
)
ROLES = ("target_test", "ood_test", "retention_test")


def matrix(state, role):
    a = state["artifacts"]
    base = a["parent"][role]
    h1 = np.stack([a[f"m1_{k}"][role] - base for k in range(3)])
    h2 = np.stack(
        [
            np.mean([a[f"m2_{k}_{d}"][role] for d in range(3)], axis=0) - base
            for k in range(3)
        ]
    )
    mass = np.asarray(state["actions"]["forced"])
    u1, u2 = mass @ h1, mass @ h2
    return np.stack(
        [
            u1,
            u2,
            h1.mean(0),
            h2.mean(0),
            u2 - u1,
            (h2 - h1).mean(0),
            np.asarray(state["actions"]["replay"]) @ h2,
        ]
    )


def bootstrap(states, draws=5000):
    if len(states) != 16 or {s["stream"] for s in states} != {
        "smollm3-3b/code",
        "smollm3-3b/math",
    }:
        raise ValueError("Complete balanced sixteen-parent extension required")
    if any(
        sum(s["stream"] == g for s in states) != 8
        for g in {s["stream"] for s in states}
    ):
        raise ValueError("Eight parents per task required")
    rng = np.random.default_rng(20261024)
    weights = np.zeros((draws, 16))
    for group in sorted({s["stream"] for s in states}):
        ids = np.array([i for i, s in enumerate(states) if s["stream"] == group])
        chosen = rng.choice(ids, size=(draws, 8))
        for i in ids:
            weights[:, i] = (chosen == i).sum(1) / 16
    point = np.zeros(len(NAMES))
    samples = np.zeros((draws, len(NAMES)))
    shared = {}
    head_point = {r: np.zeros(len(NAMES)) for r in ROLES}
    head_boot = {r: np.zeros_like(samples) for r in ROLES}
    stream_point = {s["stream"]: np.zeros(len(NAMES)) for s in states}
    stream_boot = {s["stream"]: np.zeros_like(samples) for s in states}
    for i, state in enumerate(states):
        for role in ROLES:
            values = matrix(state, role)
            identity = tuple(tuple(x) for x in state["ids"][role])
            key = (role, identity)
            if key not in shared:
                shared[key] = rng.integers(len(identity), size=(draws, len(identity)))
            mean = values.mean(1)
            sampled = values[:, shared[key]].mean(2).T
            point += mean / 48
            samples += sampled * weights[:, i, None] / 3
            head_point[role] += mean / 16
            head_boot[role] += sampled * weights[:, i, None]
            stream_point[state["stream"]] += mean / 24
            stream_boot[state["stream"]] += sampled * weights[:, i, None] * 2 / 3
    return point, samples, head_point, head_boot, stream_point, stream_boot


def summarize(states, destination, metadata, draws=5000):
    point, boot, hp, hb, sp, sb = bootstrap(states, draws)

    def intervals(p, b):
        return {
            name: {
                "mean": float(p[i]),
                "ci95": np.quantile(b[:, i], [0.025, 0.975]).tolist(),
                "units": "accuracy_fraction",
            }
            for i, name in enumerate(NAMES)
        }

    report = {
        **metadata,
        "declaration": DECLARATION,
        "parents": len(states),
        "primary": intervals(point, boot)["selected_change"],
        "aggregate": intervals(point, boot),
        "by_role": {r: intervals(hp[r], hb[r]) for r in ROLES},
        "by_task_descriptive": {g: intervals(sp[g], sb[g]) for g in sp},
        "terminal_draws": sum(s["terminal_draws"] for s in states),
        "continuation_draws": 144,
        "scored_checkpoints": 208,
        "forecaster_fits": 0,
        "automatic_expansion": False,
        "decision": "THIRD_FAMILY_EXTENSION_COMPLETE_REVIEW_WITH_ORIGINAL_CONFIRMATION",
        "limitations": "One additional fixed3B model; eight initial runs per task; conditional on frozen development selector and finite item pools. No causal model-size claim, learned policy, five-model completion or T5 extension.",
    }
    write_json(Path(destination) / "report.json", report)
    lines = [
        "# SmolLM3 sixteen-parent extension",
        "",
        "Complete balanced model extension; no forecaster fitted.",
        "",
        "| Endpoint | Mean (pp) | Joint95% interval (pp) |",
        "| --- | ---: | --- |",
    ]
    for name, v in report["aggregate"].items():
        lines.append(
            f"| {name} | {v['mean'] * 100:+.3f} | [{v['ci95'][0] * 100:+.3f}, {v['ci95'][1] * 100:+.3f}] |"
        )
    (Path(destination) / "findings.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report
