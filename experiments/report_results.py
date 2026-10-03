"""Export auditable policy tables, paired tests, and publication figures."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from third_eye.io import write_json, file_digest
from third_eye.statistics.inference import bootstrap, paired_comparison, holm, mcnemar
from third_eye.statistics.report import trajectory_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs", nargs="+", required=True, help="Completed online run directories"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", default="greedy_h1")
    p.add_argument(
        "--final-evaluations",
        nargs="*",
        default=[],
        help="Sealed final evaluation directories",
    )
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        p.error("Result reports are immutable; choose a new directory")
    runs = []
    rows = []
    keyed = {}
    for name in a.runs:
        directory = Path(name)
        run = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if (
            metadata.get("pilot")
            or metadata["config"]["protocol"]["status"] != "frozen"
        ):
            p.error("Pilot runs cannot enter final policy comparisons")
        # Matching excludes policy but includes base revision and every shared
        # training/feature budget. Different split versions are never pooled as pairs.
        match = (
            run["model"],
            run["manifest_hash"],
            run["seed"],
            metadata["resolved_revision"],
            (metadata.get("model_source") or {}).get("proof_sha256"),
            json.dumps(metadata["config"]["training"], sort_keys=True),
            json.dumps(metadata["config"]["protocol"], sort_keys=True),
        )
        key = (match, run["policy"])
        if key in keyed:
            p.error(
                "Duplicate policy/seed trajectory; do not count reruns as independent units"
            )
        values = trajectory_metrics(run)
        if (
            run["points"][0]["generation"] != 0
            or run["points"][-1]["generation"]
            != metadata["config"]["protocol"]["depth"]
        ):
            p.error(
                "Policy reports require a complete baseline-to-frozen-depth trajectory, including stitched resume history"
            )
        keyed[key] = values
        rows.append(
            {
                "model": run["model"],
                "manifest_hash": run["manifest_hash"],
                "seed": run["seed"],
                "policy": run["policy"],
                "source": str(directory),
                **values,
            }
        )
        runs.append(run)
    out.mkdir(parents=True)
    with (out / "trajectories.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    per_block = {}
    for model, manifest in sorted({(r["model"], r["manifest_hash"]) for r in rows}):
        block = {}
        for policy in sorted({r["policy"] for r in rows}):
            selected = [
                r
                for r in rows
                if r["model"] == model
                and r["manifest_hash"] == manifest
                and r["policy"] == policy
            ]
            if selected:
                block[policy] = {
                    "seeds": sorted(r["seed"] for r in selected),
                    "three_seeds_complete": len({r["seed"] for r in selected}) >= 3,
                    "final_gain": bootstrap([r["final_gain"] for r in selected]),
                    "retention_delta": bootstrap(
                        [r["retention_delta"] for r in selected]
                    ),
                }
        per_block[model + ":" + manifest] = block
    for policy in sorted({r["policy"] for r in rows}):
        selected = [r for r in rows if r["policy"] == policy]
        aggregate[policy] = {
            key: bootstrap([r[key] for r in selected])
            for key in (
                "final_gain",
                "auc_gain",
                "harmful_update_rate",
                "retention_delta",
                "maximum_forgetting",
                "wall_gpu_hours",
            )
        }
    comparisons = {}
    for policy in sorted({r["policy"] for r in rows} - {a.baseline}):
        matches = [
            match
            for match, pol in keyed
            if pol == policy and (match, a.baseline) in keyed
        ]
        if matches:
            comparisons[policy] = paired_comparison(
                [keyed[(m, policy)]["final_gain"] for m in matches],
                [keyed[(m, a.baseline)]["final_gain"] for m in matches],
            )
    correction = holm(
        {pol: v["paired_permutation_p"] for pol, v in comparisons.items()}
    )
    for pol, adjusted in correction.items():
        comparisons[pol]["holm_adjusted_p"] = adjusted
    final_groups = defaultdict(dict)
    final_scores = {}
    final_rows = []
    for name in a.final_evaluations:
        path = Path(name)
        score = json.loads((path / "scores.json").read_text(encoding="utf-8"))
        items = [
            json.loads(line)
            for line in (path / "items.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        group = (
            score["model"],
            score["seed"],
            score["final_manifest_hash"],
            score["config_hash"],
            score["resolved_revision"],
            (score.get("model_source") or {}).get("proof_sha256"),
        )
        if (group, score["policy"]) in final_scores:
            p.error("Duplicate final evaluation policy/seed")
        macro = float(
            np.dot(
                score["utility_weights"],
                [
                    score["scores"][role]
                    for role in ("target_test", "ood_test", "retention_test")
                ],
            )
        )
        final_scores[(group, score["policy"])] = macro
        final_rows.append(
            {
                "model": score["model"],
                "seed": score["seed"],
                "final_manifest_hash": score["final_manifest_hash"],
                "policy": score["policy"],
                "macro": macro,
                **score["scores"],
            }
        )
        for role in ("target_test", "ood_test", "retention_test"):
            final_groups[(*group, role)][score["policy"]] = {
                r["id"]: r["passed"] for r in items if r["role"] == role
            }
    item_tests = []
    for group, policies in final_groups.items():
        if a.baseline not in policies:
            continue
        baseline = policies[a.baseline]
        for policy, items in policies.items():
            if policy == a.baseline:
                continue
            if set(items) != set(baseline):
                p.error("Final item tests require identical paired IDs")
            ids = sorted(items)
            item_tests.append(
                {
                    "model": group[0],
                    "seed": group[1],
                    "policy": policy,
                    "role": group[-1],
                    **mcnemar([items[i] for i in ids], [baseline[i] for i in ids]),
                }
            )
    if final_rows:
        with (out / "final_benchmarks.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
            writer.writeheader()
            writer.writerows(final_rows)
    final_comparisons = {}
    for policy in sorted({row["policy"] for row in final_rows} - {a.baseline}):
        matches = [
            group
            for group, pol in final_scores
            if pol == policy and (group, a.baseline) in final_scores
        ]
        if matches:
            final_comparisons[policy] = paired_comparison(
                [final_scores[(g, policy)] for g in matches],
                [final_scores[(g, a.baseline)] for g in matches],
            )
    for policy, value in holm(
        {pol: r["paired_permutation_p"] for pol, r in final_comparisons.items()}
    ).items():
        final_comparisons[policy]["holm_adjusted_p"] = value
    for index, value in holm(
        {str(i): r["exact_p"] for i, r in enumerate(item_tests)}
    ).items():
        item_tests[int(index)]["holm_adjusted_p"] = value
    write_json(
        out / "report.json",
        {
            "evaluation": "development trajectories; final scores remain separately identified",
            "aggregate": aggregate,
            "per_model_task_block": per_block,
            "paired_final_gain_comparisons": comparisons,
            "baseline": a.baseline,
            "final_item_tests": item_tests,
            "final_benchmark_macro_comparisons": final_comparisons,
            "final_benchmark_rows": final_rows,
            "required_core_seeds": 3,
            "inputs": {
                str(Path(n) / "trajectory.json"): file_digest(
                    Path(n) / "trajectory.json"
                )
                for n in a.runs
            },
        },
    )
    lines = [
        "# Third Eye measured policy results",
        "",
        "Development trajectories; benchmark test scores are evaluated separately.",
        "",
        "| Policy | Final gain (points) | Retention change (points) | Harmful updates | GPU wall hours |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pol, v in aggregate.items():
        lines.append(
            f"| {pol} | {100 * v['final_gain']['mean']:.3f} | {100 * v['retention_delta']['mean']:.3f} | {100 * v['harmful_update_rate']['mean']:.2f}% | {v['wall_gpu_hours']['mean']:.3f} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not a.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for policy in aggregate:
            selected = [r for r in runs if r["policy"] == policy]
            lengths = {len(r["points"]) for r in selected}
            if len(lengths) != 1:
                raise ValueError("Plot trajectories with matched depths only")
            values = np.asarray(
                [
                    [
                        100 * (p["utility"] - r["points"][0]["utility"])
                        for p in r["points"]
                    ]
                    for r in selected
                ]
            )
            x = np.arange(values.shape[1])
            ax.plot(x, values.mean(axis=0), marker="o", label=policy)
            if len(values) > 1:
                ax.fill_between(
                    x,
                    values.mean(axis=0) - values.std(axis=0, ddof=1),
                    values.mean(axis=0) + values.std(axis=0, ddof=1),
                    alpha=0.12,
                )
        ax.set(xlabel="Accepted generation", ylabel="Development macro gain (points)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "trajectories.pdf")
        fig.savefig(out / "trajectories.png", dpi=200)
        plt.close(fig)
    print(f"Measured result artifacts: {out}")


if __name__ == "__main__":
    main()
