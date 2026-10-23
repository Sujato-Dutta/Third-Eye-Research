"""Frozen parent/shared-item uncertainty and writing artifacts for final cohort."""

import csv
from pathlib import Path

import numpy as np

from third_eye.experiments.confirmation import DECLARED
from third_eye.io import write_json
from third_eye.statistics.inference import holm


ROLES = ("target_test", "ood_test", "retention_test")
NAMES = DECLARED["policies"] + ["forced_immediate", "model_interaction"]


def values(state, role):
    artifacts = state["artifacts"]
    parent = artifacts["parent"][role]
    future = np.stack(
        [
            np.mean([artifacts[f"m2_{k}_{d}"][role] for d in range(3)], axis=0) - parent
            for k in range(3)
        ]
    )
    immediate = np.stack([artifacts[f"m1_{k}"][role] - parent for k in range(3)])
    policy = np.stack(
        [np.asarray(state["actions"][name]) @ future for name in DECLARED["policies"]]
    )
    h1 = np.asarray(state["actions"]["forced"]) @ immediate
    sign = 2 if state["stream"].startswith("qwen") else -2
    return np.concatenate([policy, h1[None], (sign * policy[1])[None]], axis=0)


def bootstrap(states, draws=5000, mode="joint"):
    if mode not in {"parent", "item", "joint"} or len(states) % 4:
        raise ValueError("Balanced cohort and declared resampling mode required")
    streams = sorted({state["stream"] for state in states})
    if len(streams) != 4 or any(
        sum(s["stream"] == stream for s in states) != len(states) // 4
        for stream in streams
    ):
        raise ValueError("Complete balanced four-stream sample required")
    rng = np.random.default_rng(20261022)
    weights = np.ones((draws, len(states))) / len(states)
    if mode != "item":
        weights[:] = 0
        for stream in streams:
            group = np.array(
                [i for i, state in enumerate(states) if state["stream"] == stream]
            )
            selected = rng.choice(group, size=(draws, len(group)))
            for i in group:
                weights[:, i] = (selected == i).sum(1) / len(states)
    shared, sets = {}, {}
    result = np.zeros((draws, len(NAMES)))
    retention = np.zeros((draws, 4))
    point = np.zeros(len(NAMES))
    rpoint = np.zeros(4)
    subgroup = {stream: np.zeros((draws, len(NAMES))) for stream in streams}
    subgroup_point = {stream: np.zeros(len(NAMES)) for stream in streams}
    for i, state in enumerate(states):
        for role in ROLES:
            identity = tuple(tuple(value) for value in state["ids"][role])
            key = (role, identity)
            for other in sets.get(role, []):
                if set(other) & set(identity) and other != identity:
                    raise ValueError(
                        "Partially overlapping item pools need explicit shared-identity resampling"
                    )
            sets.setdefault(role, []).append(identity)
            matrix = values(state, role)
            point += matrix.mean(1) / (len(states) * 3)
            subgroup_point[state["stream"]] += matrix.mean(1) * 4 / (len(states) * 3)
            if mode == "parent":
                sampled = np.broadcast_to(matrix.mean(1), (draws, len(NAMES)))
            else:
                if key not in shared:
                    shared[key] = rng.integers(
                        len(identity), size=(draws, len(identity))
                    )
                sampled = matrix[:, shared[key]].mean(2).T
            result += sampled * weights[:, i, None] / 3
            subgroup[state["stream"]] += sampled * weights[:, i, None] * 4 / 3
            if role == "retention_test":
                retention += sampled[:, :4] * weights[:, i, None]
                rpoint += matrix[:4].mean(1) / len(states)
    return {
        "point": point,
        "boot": result,
        "retention_point": rpoint,
        "retention_boot": retention,
        "subgroup_point": subgroup_point,
        "subgroup_boot": subgroup,
    }


def interval(point, boot, alpha=0.05):
    return {
        "mean": float(point),
        "ci": np.quantile(boot, [alpha / 2, 1 - alpha / 2]).tolist(),
        "units": "accuracy_fraction",
    }


def summarize(states, output, metadata):
    output = Path(output)
    modes = {
        mode: bootstrap(states, DECLARED["bootstrap_draws"], mode)
        for mode in ("parent", "item", "joint")
    }
    joint = modes["joint"]
    primary = interval(joint["point"][-1], joint["boot"][:, -1])
    secondary, pvalues = {}, {}
    for left, right in DECLARED["secondary_family"]:
        left = "forced" if left == "forced_future" else left
        li, ri = NAMES.index(left), NAMES.index(right)
        mean = joint["point"][li] - joint["point"][ri]
        sample = joint["boot"][:, li] - joint["boot"][:, ri]
        key = left + "_minus_" + right
        noise = sample - mean
        pvalues[key] = float(
            (1 + np.sum(np.abs(noise) >= abs(mean) - 1e-12)) / (len(noise) + 1)
        )
        secondary[key] = {
            **interval(mean, sample),
            "simultaneous_ci95": np.quantile(
                sample, [0.05 / 12, 1 - 0.05 / 12]
            ).tolist(),
            "centered_crossed_bootstrap_p": pvalues[key],
        }
    adjusted = holm(pvalues)
    for key, value in secondary.items():
        value["holm_p"] = adjusted[key]
    policies = {
        name: {
            **interval(joint["point"][i], joint["boot"][:, i]),
            "retention": interval(
                joint["retention_point"][i], joint["retention_boot"][:, i]
            ),
            "retention_lower_one_sided95": float(
                np.quantile(joint["retention_boot"][:, i], 0.05)
            ),
            "coverage": float(
                np.mean([sum(state["actions"][name]) for state in states])
            ),
        }
        for i, name in enumerate(DECLARED["policies"])
    }
    subgroups = {
        stream: {
            name: interval(
                joint["subgroup_point"][stream][i], joint["subgroup_boot"][stream][:, i]
            )
            for i, name in enumerate(NAMES[:-1])
        }
        for stream in joint["subgroup_point"]
    }
    model = {}
    for family in ("llama", "qwen"):
        streams = [stream for stream in subgroups if stream.startswith(family)]
        point = np.mean([joint["subgroup_point"][stream][1] for stream in streams])
        boot = np.mean(
            [joint["subgroup_boot"][stream][:, 1] for stream in streams], axis=0
        )
        model[family] = interval(point, boot, alpha=0.025)
    benefit_harm = (
        primary["ci"][0] > 0
        and model["qwen"]["ci"][0] > 0
        and model["llama"]["ci"][1] < 0
    )
    rule_value = (
        all(
            secondary["family_admission_minus_" + name]["simultaneous_ci95"][0] > 0
            for name in ("hold", "forced", "replay")
        )
        and policies["family_admission"]["mean"] - max(0, policies["forced"]["mean"])
        >= 0.005
        and policies["family_admission"]["retention_lower_one_sided95"] > -0.005
    )
    report = {
        **metadata,
        "independent_parents": len(states),
        "parents_per_stream": len(states) // 4,
        "primary": primary,
        "model_difference_replicated": primary["ci"][0] > 0,
        "benefit_harm_crossover_replicated": benefit_harm,
        "family_rule_practical_advantage_supported": bool(rule_value),
        "policies": policies,
        "secondary": secondary,
        "model_means_simultaneous95": model,
        "subgroups_descriptive": subgroups,
        "uncertainty": {
            mode: {
                name: interval(value["point"][i], value["boot"][:, i])
                for i, name in enumerate(NAMES)
            }
            for mode, value in modes.items()
        },
        "terminal_draws": sum(state["terminal_draws"] for state in states),
        "total_continuation_draws": len(states) * 9,
        "forecaster_fits": 0,
        "limitations": "Two fixed backbones, two tasks, size-two LoRA correction updates, initial parents and H2. No new forecaster, architecture claim, universal harm or causal model-identity claim. Fixed policies; joint intervals condition on development-derived rule and finite pilot choices. Shared exact item pools resample together. Historical optimizer/harvest variance is not separately identified.",
        "decision": "EXPERIMENTS_COMPLETE_FREEZE_EVIDENCE_AND_WRITE",
        "automatic_expansion": False,
    }
    write_json(output / "report.json", report)
    with (output / "policies.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "policy",
                "utility_pp",
                "joint_ci95_low_pp",
                "joint_ci95_high_pp",
                "retention_pp",
                "coverage",
            ]
        )
        for name, value in policies.items():
            writer.writerow(
                [
                    name,
                    value["mean"] * 100,
                    value["ci"][0] * 100,
                    value["ci"][1] * 100,
                    value["retention"]["mean"] * 100,
                    value["coverage"],
                ]
            )
    lines = [
        "# Final empirical confirmation",
        "",
        f"Complete balanced cohort: {len(states)} independent parents, {len(states) // 4} per stream. No forecaster fitted.",
        "",
        "Primary: Qwen-minus-Llama future forced-update utility, percentage points.",
        f"Estimate {primary['mean'] * 100:.3f}; joint 95% interval [{primary['ci'][0] * 100:.3f}, {primary['ci'][1] * 100:.3f}].",
        f"Model difference replicated: {report['model_difference_replicated']}. Benefit/harm crossover replicated: {benefit_harm}.",
        f"Fixed family rule passes all declared practical comparisons: {rule_value}.",
        "",
        "| Policy | Future utility vs hold (pp) | Joint 95% interval (pp) | Coverage |",
        "| --- | ---: | --- | ---: |",
    ]
    for name, value in policies.items():
        lines.append(
            f"| {name} | {value['mean'] * 100:.3f} | [{value['ci'][0] * 100:.3f}, {value['ci'][1] * 100:.3f}] | {value['coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All secondary comparisons, Holm adjustments, model/stream estimates, retention and parent/item/joint intervals are in report.json. Scope and limitations apply whether the hypothesis is replicated or not. No follow-up collection is launched.",
            "",
            "Final allocation accounting must be reconciled against Slurm before reporting project SUs.",
        ]
    )
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(policies)
    mean = np.array([policies[name]["mean"] for name in names]) * 100
    lower = np.array([policies[name]["ci"][0] for name in names]) * 100
    upper = np.array([policies[name]["ci"][1] for name in names]) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.vlines(np.arange(4), lower, upper)
    ax.plot(np.arange(4), mean, "o")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(np.arange(4), names, rotation=15)
    ax.set_ylabel("Future utility vs hold (percentage points)")
    ax.set_title("Independent confirmation; joint parent/item 95% intervals")
    fig.tight_layout()
    fig.savefig(output / "policy_utility.pdf")
    fig.savefig(output / "policy_utility.png", dpi=200)
    plt.close(fig)
