"""Compute-only decision audit of the completed frozen admission development set."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from admission_pilots import load_pilot_states
from third_eye.cluster import require_compute_allocation
from third_eye.experiments.admission_audit import (
    DECLARED,
    evaluate,
    reproduce,
    summarize_records,
    uncertainty,
)
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    require_compute_allocation()
    plan = read(args.plan)
    if (
        plan["source_tree_sha256"] != source_inventory()["source_tree_sha256"]
        or plan["declaration"] != DECLARED
    ):
        raise ValueError("Frozen audit source/declaration differs")
    validation = read(plan["validation"])
    if (
        validation.get("passed") is not True
        or validation["source_tree_sha256"] != plan["source_tree_sha256"]
    ):
        raise ValueError("Matching full CPU validation required")
    for key in (
        "pilot_plan",
        "pilot_report",
        "pilot_fits",
        "pilot_completion",
        "pilot_accounting",
        "pilot_submissions",
    ):
        if file_digest(plan[key]) != plan[key + "_sha256"]:
            raise ValueError("Frozen development input differs: " + key)
    historical, report = read(plan["pilot_plan"]), read(plan["pilot_report"])
    if (
        historical["source_tree_sha256"] != report["source_tree_sha256"]
        or read(plan["pilot_completion"])["report_sha256"]
        != plan["pilot_report_sha256"]
    ):
        raise ValueError("Completed B source/report identity differs")
    states, coverage, hashes = load_pilot_states(historical)
    if len(states) != 16 or any(not r["complete"] for r in coverage):
        raise ValueError("All sixteen complete parents required")
    for path, sha in report["input_hashes"].items():
        if file_digest(path) != sha:
            raise ValueError("Previously reviewed B artifact changed")
    output = Path(plan["output"])
    if output.exists():
        raise FileExistsError("Audit outcomes are immutable; no silent rerun")
    output.mkdir(parents=True)
    results, records = {}, {}
    for scheme in DECLARED["cv"]:
        records[scheme] = evaluate(states, scheme)
        results[scheme] = summarize_records(states, records[scheme])
    original_reproduced = reproduce(read(plan["pilot_fits"]), records["leave_parent"])
    precision = uncertainty(states, records["leave_parent"])
    accounting, submissions = (
        read(plan["pilot_accounting"]),
        read(plan["pilot_submissions"]),
    )
    observed, limits = {}, {}
    for stream, job in submissions["pilot_arrays"].items():
        observed[stream] = [
            r["elapsed_seconds"]
            for r in accounting["scheduler_rows"]
            if r["id"].startswith(job["job"] + "_")
        ]
        if len(observed[stream]) != 4:
            raise ValueError("Four actual runtime benchmarks per stream required")
        limits[stream] = int(np.ceil(max(observed[stream]) * 1.25 / 900) * 900)
    cost = {
        "measured_max_with_25pct_margin_rounded_15min_seconds": limits,
        "remaining_project_sus_at_declaration": plan["remaining_project_sus"],
        "rows": [
            {
                "parents": n,
                "measured_mean_gpu_sus": n
                * np.mean([v for values in observed.values() for v in values])
                / 3600,
                "conservative_base_gpu_reservation_sus": (n / 4)
                * sum(limits.values())
                / 3600,
                "fits_original_560_su_C_ceiling_before_extra_scoring": (n / 4)
                * sum(limits.values())
                / 3600
                <= 560,
            }
            for n in (32, 48, 64, 96)
        ],
        "limitation": "Base collection cost only; additional/sealed-item inference has not been timed. Larger-I power scenarios are hypothetical and not fully priced. No powered/affordable confirmation declaration follows from this table.",
    }
    value = {
        "status": "exploratory_decision_review_required",
        "declaration": DECLARED,
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_sha256": file_digest(args.plan),
        "original_B_gate_fits_reproduced": original_reproduced,
        "independent_parents": 16,
        "held_seed_groups": 4,
        "coverage": coverage,
        "input_hashes": hashes,
        "policies": results,
        "precision": precision,
        "cost": cost,
        "sealed_suite_metadata_only": historical["manifest_audits"],
        "decision": "STOP_FOR_POLICY_PRECISION_AND_COST_REVIEW_NO_C_RELEASE",
        "automatic_expansion": False,
        "limitations": "B outcomes were already inspected. Context means and residual models are exploratory diagnostics, not promoted controllers. Within-parent ranks have only K=3; seed-block sensitivity has four groups; fitted-model uncertainty and future-item/domain generalization remain unresolved. No altered historical endpoint or restored forecasting claim.",
    }
    write_json(output / "report.json", value)
    write_json(output / "predictions.json", records)
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "automatic_expansion": False,
            "report_sha256": file_digest(output / "report.json"),
        },
    )
    print(
        json.dumps(
            {
                "completed": True,
                "original_fits_reproduced": original_reproduced,
                "automatic_expansion": False,
            }
        )
    )


if __name__ == "__main__":
    main()
