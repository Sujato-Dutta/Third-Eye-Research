"""Render already-computed study summaries; no fitting or model selection."""

import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs/deployment/evidence"
OUT = EVIDENCE / "signal_figures_v1"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    development = json.loads(
        (EVIDENCE / "signal_diagnostic_v1_review.json").read_text()
    )
    confirmation = json.loads(
        (EVIDENCE / "signal_confirmation_v1_review.json").read_text()
    )
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for target, label, color in [
        ("immediate", "Immediate (H=1)", "#0072B2"),
        ("future", "Future (H=2)", "#D55E00"),
        ("continuation", "Continuation increment", "#009E73"),
    ]:
        values = np.array(
            [
                [
                    development["fits"][f"gru_{target}_n{n}_seed{s}"]["own_target"][
                        "utility"
                    ]["spearman"]["state_mean"]
                    for s in [42, 43, 44]
                ]
                for n in [5, 10, 15, 20]
            ]
        )
        ax.plot([5, 10, 15, 20], values.mean(1), "o-", color=color, label=label)
        ax.fill_between(
            [5, 10, 15, 20], values.min(1), values.max(1), color=color, alpha=0.12
        )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set(
        xticks=[5, 10, 15, 20],
        xlabel="Training trajectories (nested subsets)",
        ylabel="Mean within-state Spearman",
        title="Development learning curves: fixed independent GRU",
    )
    ax.legend(frameon=False)
    fig.text(
        0.11,
        -0.02,
        "40 development states / 8 trajectories. Shading: range of 3 fitting seeds, not a confidence interval.",
        fontsize=8,
    )
    save(fig, "learning_curves")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = [
        "immediate_own_minus_future_own",
        "future_own_minus_immediate_future",
        "immediate_own_minus_continuation_own",
    ]
    labels = [
        "H1 own minus H2 own",
        "H2 minus H1 on future",
        "H1 own minus continuation own",
    ]
    for ax, report, title in zip(
        axes, [development, confirmation], ["Development", "Held-out confirmation"]
    ):
        for i, name in enumerate(names):
            r = report["paired_comparisons"][name]
            ax.plot(r["ci95"], [i, i], color="#0072B2", linewidth=2)
            ax.scatter([r["mean"]], [i], color="#0072B2", zorder=3)
            ax.annotate(
                f"Holm p={r['holm_p_three_predeclared_contrasts']:.3f}",
                (r["ci95"][1], i),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=8,
            )
        ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set(
            yticks=range(3),
            yticklabels=labels,
            xlabel="Paired difference in Spearman",
            title=title,
            ylim=(-0.5, 2.6),
            xlim=(-0.35, 0.8),
        )
        ax.invert_yaxis()
    fig.tight_layout()
    fig.text(
        0.02,
        -0.035,
        "Seed-averaged within state; paired across 8 trajectories per cohort. Unadjusted 95% bootstrap intervals; Holm-adjusted p-values.",
        fontsize=8,
    )
    save(fig, "paired_confirmation")
    summary = {}
    for cohort, report in [("development", development), ("heldout", confirmation)]:
        summary[cohort] = {
            target: {
                "mean_spearman": statistics.mean(
                    report["fits"][f"gru_{target}_n20_seed{s}"]["own_target"][
                        "utility"
                    ]["spearman"]["state_mean"]
                    for s in [42, 43, 44]
                ),
                "mean_top1": statistics.mean(
                    report["fits"][f"gru_{target}_n20_seed{s}"]["own_target"][
                        "utility"
                    ]["top1"]["state_mean"]
                    for s in [42, 43, 44]
                ),
            }
            for target in ["immediate", "future", "continuation"]
        }
    (OUT / "displayed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
