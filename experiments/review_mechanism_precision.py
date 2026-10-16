"""Exploratory mechanism diagnosis and precision planning; no new GPU outcomes."""

import argparse
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strengthen_evidence import artifacts, read
from forecasting_pilots import inspect_task
from third_eye.cluster import require_compute_allocation
from third_eye.data.splits import load_manifest
from third_eye.experiments.mechanism import choose, folds, matrix
from third_eye.forecasting.continuation import repeated_targets
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory


def panel(h1, control, intervention):
    """Future outcomes cannot change the fixed immediate candidate choice."""
    h1, control, intervention = [
        np.asarray(x, float) for x in (h1, control, intervention)
    ]
    if (
        h1.shape != (3, 3)
        or control.shape != (3, 3, 3)
        or intervention.shape != (3, 3, 3)
    ):
        raise ValueError("Require K=3, three draws and three heads")
    if not all(np.isfinite(x).all() for x in (h1, control, intervention)):
        raise ValueError("Finite matched outcomes required")
    mass = choose(h1.mean(1))
    effects = intervention - control
    return dict(
        selected_mass=mass.tolist(),
        mean_all_candidate_effect_by_head=effects.mean((0, 1)).tolist(),
        same_selected_effect_by_head=(mass @ effects.mean(0)).tolist(),
        same_selected_effect_by_draw=(effects.mean(2) @ mass).tolist(),
        immediate_delta_by_candidate=h1.mean(1).tolist(),
        future_effect_by_candidate=effects.mean((0, 2)).tolist(),
    )


def precision(std):
    """Normal approximation is planning only, not a powered confirmation claim."""
    if not math.isfinite(std) or std < 0:
        raise ValueError("Finite nonnegative parent-level standard deviation required")
    z95, z80 = NormalDist().inv_cdf(0.975), NormalDist().inv_cdf(0.8)
    return dict(
        pilot_parent_sd=std,
        normal_approximation_80pct_power=[
            dict(
                effect_pp=effect * 100,
                parents=max(4, 4 * math.ceil((((z95 + z80) * std / effect) ** 2) / 4)),
            )
            for effect in (0.005, 0.01, 0.02)
        ],
        approximate_ci_half_width=[
            dict(parents=n, half_width_pp=100 * z95 * std / math.sqrt(n))
            for n in (16, 24, 32, 48, 64)
        ],
        caution="Eight developmental parents provide uncertain variance estimates. Normal approximation assumes independent balanced parent effects, fixed variance and no multiplicity adjustment. It does not establish power for T=5 online trajectories, fresh item pools or new harvest regimes.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_compute_allocation()
    campaign, output = Path(args.campaign), Path(args.output)
    if output.exists():
        parser.error("Never overwrite a previous diagnostic")
    report_path = campaign / "evidence/report.json"
    if file_digest(report_path) != args.report_sha256:
        parser.error("Frozen first-wave report changed")
    report, plan = read(report_path), read(campaign / "plan.json")
    if (
        report["missing"]
        or report["completed_states"] != 8
        or report["development_only"] is not True
    ):
        parser.error("Require eight complete developmental parents")
    pilot = read(plan["pilot_plan"])
    if file_digest(plan["pilot_plan"]) != plan["pilot_plan_sha256"]:
        parser.error("Native pilot plan changed")
    states, hashes, norms = [], {str(report_path): file_digest(report_path)}, []
    for index, task in enumerate(pilot["tasks"]):
        original = inspect_task(
            task, pilot["continuation_seed_offsets"], pilot["source_tree_sha256"]
        )[0]
        receipt_path = campaign / f"evidence/task_{index}/completed.json"
        receipt = read(receipt_path)
        if not receipt["passed"] or receipt["plan_hash"] != digest(plan):
            raise ValueError("Intervention receipt differs from frozen plan")
        rows = []
        for original_row in original:
            path = (
                campaign
                / f"evidence/task_{index}"
                / original_row["candidate_id"]
                / "intervention_record.json"
            )
            row = read(path)
            hashes[str(path)] = file_digest(path)
            if (
                row["artifact_kind"] != "paired_candidate_lr_intervention"
                or row["batch_hash"] != original_row["batch_hash"]
                or row["parent_adapter_hash"] != original_row["parent_adapter_hash"]
            ):
                raise ValueError(
                    "Intervention no longer matches its parent/data control"
                )
            if not row["intervention"]["forbidden_for_forecaster_training"]:
                raise ValueError(
                    "Baseline-LR probes must not become intervention training inputs"
                )
            rows.append(row)
            norms.append(
                dict(
                    stream=task["stream"],
                    trajectory_id=row["trajectory_id"],
                    candidate_id=row["candidate_id"],
                    control=row["intervention"]["control_adapter_delta_norm"],
                    half_lr=row["intervention"]["intervention_adapter_delta_norm"],
                )
            )
            for key in ("parent", "t1"):
                path = row["evaluation_artifacts"][key]
                hashes[path] = file_digest(path)
            for draw in row["continuation_replicates"]:
                hashes[draw["items_path"]] = file_digest(draw["items_path"])
        repeated_targets(rows)
        _, splits = load_manifest(task["manifest"])
        parent, h1, baseline = artifacts(original, splits)
        _, _, small = artifacts(rows, splits)
        for row in original:
            hashes[row["evaluation_artifacts"]["t1"]] = file_digest(
                row["evaluation_artifacts"]["t1"]
            )
            for draw in row["continuation_replicates"]:
                hashes[draw["items_path"]] = file_digest(draw["items_path"])
        assignment = folds(parent)
        panels = []
        for fold in (0, 1):
            panels.append(
                panel(
                    matrix(h1, parent, assignment, fold),
                    [matrix(draw, parent, assignment, 1 - fold) for draw in baseline],
                    [matrix(draw, parent, assignment, 1 - fold) for draw in small],
                )
            )
        states.append(
            dict(
                stream=task["stream"],
                trajectory_id=original[0]["trajectory_id"],
                fold_panels=panels,
                mean_all_effect_by_head=np.mean(
                    [p["mean_all_candidate_effect_by_head"] for p in panels], axis=0
                ).tolist(),
                same_selected_effect_by_head=np.mean(
                    [p["same_selected_effect_by_head"] for p in panels], axis=0
                ).tolist(),
            )
        )
    groups = {}
    for stream in sorted({s["stream"] for s in states}):
        group = [s for s in states if s["stream"] == stream]
        if len(group) != 2:
            raise ValueError("Balanced two-parent streams required")
        groups[stream] = dict(
            parents=2,
            mean_all_effect_by_head=np.mean(
                [s["mean_all_effect_by_head"] for s in group], axis=0
            ).tolist(),
            same_selected_effect_by_head=np.mean(
                [s["same_selected_effect_by_head"] for s in group], axis=0
            ).tolist(),
        )
    result = dict(
        exploratory_posthoc=True,
        development_only=True,
        automatic_expansion=False,
        gate2_passed=None,
        primary_first_wave_unchanged=report["comparisons"][
            "half_lr_same_candidate_minus_baseline"
        ],
        heads=["target", "ood", "retention"],
        states=states,
        descriptive_streams=groups,
        adapter_norm_manipulation=dict(
            candidates=len(norms),
            smaller=sum(n["half_lr"] < n["control"] for n in norms),
            records=norms,
        ),
        precision_planning={
            key: precision(report["comparisons"][key]["std"])
            for key in (
                "half_lr_same_candidate_minus_baseline",
                "half_lr_mean_all_draws_delta",
                "guarded_minus_immediate",
                "half_lr_reselected_minus_baseline",
            )
        },
        budget_planning=dict(
            maximum_reserved_fresh_24h_single_node_tasks_in_384_sus=16,
            remaining_stage_semantics_unfrozen=True,
            caution="A task is not automatically an independent trajectory. Paired LR replication can require both control/intervention collection; use measured costs before allocating task slots. Do not multiply folds, candidates or draws into the sample size.",
        ),
        interpretation="Diagnosis describes how average intervention effects differ from selection-weighted effects. It supplies no corrected causal discovery, forecasting advantage, safe policy or confirmation release. Earlier primary endpoints and multiplicity family remain intact.",
        input_sha256=hashes,
        **source_inventory(),
    )
    write_json(output / "report.json", result)
    write_json(
        output / "completed.json",
        dict(
            passed=True,
            exploratory_posthoc=True,
            report_sha256=file_digest(output / "report.json"),
            **source_inventory(),
        ),
    )


if __name__ == "__main__":
    main()
