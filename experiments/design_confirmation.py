"""Source-bound allocated-CPU confirmation design; no fresh outcome collection."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admission_pilots import load_pilot_states
from third_eye.cluster import require_compute_allocation
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import load_final_manifest
from third_eye.experiments.admission import views
from third_eye.experiments.confirmation_design import (
    DECLARATION,
    bootstrap,
    choose_design,
    conditional_power,
    policy_actions,
    price,
)
from third_eye.experiments.mechanism import choose, folds, ROLES
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    require_compute_allocation()
    plan = read(args.plan)
    source = source_inventory()["source_tree_sha256"]
    receipt = read(plan["validation"])
    if (
        plan["source_tree_sha256"] != source
        or plan["declaration"] != DECLARATION
        or receipt.get("passed") is not True
        or receipt["source_tree_sha256"] != source
    ):
        raise ValueError(
            "Matching source, declaration and full CPU validation required"
        )
    for key, value in plan["inputs"].items():
        if file_digest(value["path"]) != value["sha256"]:
            raise ValueError("Bound evidence changed: " + key)
    paths = {key: value["path"] for key, value in plan["inputs"].items()}
    pilot = read(paths["pilot_plan"])
    report = read(paths["audit_report"])
    classification = read(paths["classification_report"])
    if (
        classification["remaining_flagged_artifacts"]
        or classification["original_unreadable_artifacts"]
    ):
        raise ValueError("Recorded exposure is unresolved")
    if (
        read(paths["classification_completion"])["report_sha256"]
        != plan["inputs"]["classification_report"]["sha256"]
    ):
        raise ValueError("Classification receipt differs")
    states, coverage, hashes = load_pilot_states(pilot)
    if len(states) != 16 or any(not row["complete"] for row in coverage):
        raise ValueError("Complete fresh B cohort required")
    for path, sha in read(paths["pilot_report"])["input_hashes"].items():
        if file_digest(path) != sha:
            raise ValueError("Reviewed B artifact changed")
    records = read(paths["audit_predictions"])["leave_parent"]
    actions = policy_actions(records)
    for index, state in enumerate(states):
        h1, _, _ = views(state["panel"], folds(state["parent"]), 0)
        immediate = choose(h1.mean(1))
        actions[index, 0] = immediate
        actions[index, 1] = immediate * (
            (h1[:, 0] > 0) & (h1[:, 1] >= 0) & (h1[:, 2] >= 0)
        )
    suite_sizes, suite_provenance, calibration = {}, {}, None
    for selection, audit in pilot["manifest_audits"].items():
        if (
            file_digest(selection) != audit["sha256"]
            or file_digest(audit["reserved_final_manifest"])
            != audit["reserved_final_manifest_sha256"]
        ):
            raise ValueError("Current suite provenance changed")
        _, dev = load_manifest(selection)
        _, final = load_final_manifest(audit["reserved_final_manifest"], selection)
        task = dev["train"][0].task
        suite_sizes[task] = {
            role.replace("_test", "_dev"): len(examples)
            for role, examples in final.items()
        }
        suite_provenance[task] = audit
        if task == "code":
            ordered = sorted(
                final["ood_test"],
                key=lambda ex: digest(
                    ["third-eye-cost-calibration-20261022", ex.id, ex.prompt_hash]
                ),
            )
            calibration = {
                "purpose": "cost_only_excluded_from_confirmation_scoring",
                "original_final_manifest": audit["reserved_final_manifest"],
                "original_final_sha256": audit["reserved_final_manifest_sha256"],
                "items": [
                    {"id": ex.id, "prompt_sha256": ex.prompt_hash} for ex in ordered[:8]
                ],
                "no_accuracy_or_method_selection": True,
            }
            suite_sizes[task]["ood_dev"] -= 8
    if calibration is None or len(calibration["items"]) != 8:
        raise ValueError("Fixed code cost-calibration reservation required")
    benchmarks = [read(paths[f"benchmark_{i}"]) for i in range(4)]
    base_limits = report["cost"]["measured_max_with_25pct_margin_rounded_15min_seconds"]
    method_promising = all(
        report["policies"][scheme]["ridge_h1"]["metrics"]["minus_hold"]["mean"] >= 0.005
        for scheme in ("leave_parent", "leave_seed")
    )
    rows = []
    for parents in DECLARATION["parents"]:
        for size in DECLARATION["target_ood_sizes"]:
            item_counts = {
                task + "/" + role: min(
                    256 if role == "retention_dev" else size, suite_sizes[task][role]
                )
                for task in ("math", "code")
                for role in ROLES
            }
            point, boot, retention_point, retention_boot = bootstrap(
                states,
                actions,
                parents,
                item_counts,
                draws=DECLARATION["bootstrap_draws"],
            )
            sensitivity = []
            for shrinkage in DECLARATION["effect_shrinkage"]:
                for inflation in DECLARATION["sd_inflation"]:
                    sensitivity.append(
                        {
                            "effect_shrinkage": shrinkage,
                            "sd_inflation": inflation,
                            **conditional_power(point, boot, shrinkage, inflation),
                        }
                    )
            rows.append(
                {
                    "parents": parents,
                    "requested_target_ood_size": size,
                    "item_counts": item_counts,
                    "pilot_orientation0_utility_pp": {
                        name: float(point[i] * 100)
                        for i, name in enumerate(DECLARATION["policies"])
                    },
                    "pilot_orientation0_model_interaction_pp": float(point[6] * 100),
                    "retention_point_pp": (retention_point * 100).tolist(),
                    "retention_conditional_sd_pp": (
                        retention_boot.std(0, ddof=1) * 100
                    ).tolist(),
                    "sensitivity": sensitivity,
                    "cost": price(parents, item_counts, base_limits, benchmarks),
                }
            )
    selected = choose_design(
        rows, method_promising, plan["remaining_sus_at_declaration"]
    )
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    value = {
        "source_tree_sha256": source,
        "declaration": DECLARATION,
        "inputs": plan["inputs"],
        "coverage": coverage,
        "input_hashes": hashes,
        "original_h1_method_promising_at_practical_effect": method_promising,
        "suite_provenance": suite_provenance,
        "calibration_reservation": calibration,
        "rows": rows,
        "selected_design_candidate": selected,
        "decision": "REVIEW_DESIGN_AND_DIRECT_CODE_COST_BEFORE_FRESH_CONFIRMATION"
        if selected
        else "NO_AFFORDABLE_POWERED_DESIGN_STOP_GPU_EXPANSION",
        "automatic_expansion": False,
        "limitations": DECLARATION["limitations"],
        "no_untouched_outcomes_read_or_generated": True,
    }
    write_json(output / "report.json", value)
    write_json(output / "code_calibration.json", calibration)
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "report_sha256": file_digest(output / "report.json"),
            "automatic_expansion": False,
        },
    )
    print(
        json.dumps(
            {
                "completed": True,
                "decision": value["decision"],
                "purpose": selected["purpose"] if selected else None,
                "automatic_expansion": False,
            }
        )
    )


if __name__ == "__main__":
    main()
