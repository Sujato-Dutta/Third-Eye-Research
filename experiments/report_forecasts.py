"""Compare held-out H=2 forecasts, ablations and trajectory-clustered ranking."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from third_eye.io import write_json, file_digest
from third_eye.statistics.inference import paired_comparison, holm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument(
        "--baseline", required=True, help="Reference saved forecaster directory"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        p.error("Forecast reports are immutable")
    baseline = Path(a.baseline).resolve()
    models = {}
    for name in a.models:
        path = Path(name).resolve()
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))[
            "test"
        ]["future_h2"]
        audit = json.loads((path / "split_audit.json").read_text(encoding="utf-8"))[
            "test"
        ]
        models[path] = (
            meta,
            metrics,
            {r["state_id"]: r["trajectory_id"] for r in audit},
        )
    if baseline not in models:
        p.error("Baseline must be included in --models")
    baseline_meta, baseline_metrics, _ = models[baseline]
    rows = []
    comparisons = {}
    for path, (meta, metrics, trajectories) in models.items():
        if (
            meta["split_hashes"]["test"] != baseline_meta["split_hashes"]["test"]
            or meta["utility_weights"] != baseline_meta["utility_weights"]
        ):
            p.error(
                "Forecast comparisons require identical held-out states and utility weights"
            )
        row = {
            "name": path.name,
            "kind": meta["kind"],
            "training_horizon": meta["horizon"],
            "seed": meta["seed"],
            "ablations": ",".join(meta["ablations"]),
            "scalar": meta["scalar"],
            "states": metrics["states"],
            **metrics["ranking"],
            "utility_mae": metrics["utility"]["mae"],
            "utility_rmse": metrics["utility"]["rmse"],
        }
        for head in ("target", "ood", "retention"):
            row[head + "_mae"] = (
                metrics["heads"][head]["mae"] if metrics["heads"] else None
            )
            row[head + "_sign_accuracy"] = (
                metrics["heads"][head]["sign_accuracy"] if metrics["heads"] else None
            )
        rows.append(row)
        if path != baseline:
            clustered = defaultdict(list)
            for state, value in metrics["per_state"].items():
                if state not in baseline_metrics["per_state"]:
                    p.error("Forecast state IDs do not match")
                clustered[trajectories[state]].append(
                    (value["top1"], baseline_metrics["per_state"][state]["top1"])
                )
            comparisons[path.name] = paired_comparison(
                [np.mean([v[0] for v in group]) for group in clustered.values()],
                [np.mean([v[1] for v in group]) for group in clustered.values()],
            )
    adjusted = holm(
        {name: r["paired_permutation_p"] for name, r in comparisons.items()}
    )
    for name, pvalue in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = pvalue
    out.mkdir(parents=True)
    with (out / "forecasting.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out / "report.json",
        {
            "evaluation_horizon": 2,
            "baseline": baseline.name,
            "rows": rows,
            "paired_top1_comparisons": comparisons,
            "independent_units": "recursive trajectories",
            "input_hashes": {
                str(path): file_digest(path / "weights.pt") for path in models
            },
        },
    )
    if not a.no_plots and not baseline_meta["scalar"]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        predictions = json.loads(
            (baseline / "test_predictions.json").read_text(encoding="utf-8")
        )
        truth = np.asarray([r["truth_h2"] for r in predictions]) * 100
        predicted = np.asarray([r["prediction"] for r in predictions]) * 100
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for i, head in enumerate(("Target", "OOD", "Retention")):
            ax = axes[i]
            ax.scatter(truth[:, i], predicted[:, i], s=10, alpha=0.45)
            bounds = [
                float(min(truth[:, i].min(), predicted[:, i].min())),
                float(max(truth[:, i].max(), predicted[:, i].max())),
            ]
            if bounds[0] == bounds[1]:
                bounds = [bounds[0] - 1, bounds[1] + 1]
            ax.plot(bounds, bounds, color="black", linewidth=1)
            ax.set(
                title=head,
                xlabel="True H=2 change (points)",
                ylabel="Predicted change (points)",
            )
            ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(out / "consequences.pdf")
        fig.savefig(out / "consequences.png", dpi=200)
        plt.close(fig)
    print(f"Forecast comparison artifacts: {out}")


if __name__ == "__main__":
    main()
