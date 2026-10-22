"""Append a checked synthetic-fixture classification to the immutable exposure audit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.cluster import require_compute_allocation
from third_eye.experiments.scoring_precision import synthetic_fixture_records
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
    source = source_inventory()["source_tree_sha256"]
    validation = read(plan["validation"])
    if (
        plan["source_tree_sha256"] != source
        or validation.get("passed") is not True
        or validation["source_tree_sha256"] != source
    ):
        raise ValueError("Matching full CPU validation required")
    for key in ("report", "completion", "inventory"):
        if file_digest(plan[key]) != plan[key + "_sha256"]:
            raise ValueError("Bound original audit changed")
    report = read(plan["report"])
    if read(plan["completion"])["report_sha256"] != plan["report_sha256"]:
        raise ValueError("Original completion differs")
    inventory = {row["path"]: row["sha256"] for row in read(plan["inventory"])}
    paths = set(report["opaque_final_evaluations"])
    for suite in report["suites"].values():
        for role in suite["roles"].values():
            for item in role["items"]:
                paths.update(item["recorded_exposure_files"])
    classified, unresolved = [], []
    for name in sorted(paths):
        path = Path(name)
        if inventory.get(name) != file_digest(path):
            raise ValueError("Scanned artifact changed")
        records = (
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else [read(path)]
        )
        validation_root = next(
            (parent.parent for parent in path.parents if parent.name == "fixtures"),
            None,
        )
        receipt_path = validation_root / "completed.json" if validation_root else None
        receipt = read(receipt_path) if receipt_path and receipt_path.is_file() else {}
        if receipt.get("passed") is True and synthetic_fixture_records(path, records):
            classified.append(
                {
                    "path": name,
                    "sha256": inventory[name],
                    "validation_receipt": str(receipt_path),
                    "validation_receipt_sha256": file_digest(receipt_path),
                    "reason": "known_mock_content_in_completed_validation_fixture",
                }
            )
        else:
            unresolved.append(name)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "source_tree_sha256": source,
        "original_report_sha256": plan["report_sha256"],
        "original_report_preserved": True,
        "classified_synthetic_artifacts": classified,
        "remaining_flagged_artifacts": unresolved,
        "original_unreadable_artifacts": report["unreadable_artifacts"],
        "status": "no_recorded_research_exposure_in_searched_artifacts"
        if not unresolved and not report["unreadable_artifacts"]
        else "recorded_exposure_unresolved",
        "limitation": "Scope-limited recorded-artifact classification, not proof against deleted/external/unlogged/pretraining exposure. No change to final item pools, policies, labels or queued benchmark jobs. No confirmation release.",
        "automatic_expansion": False,
    }
    write_json(output / "report.json", result)
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
                "classification_complete": True,
                "status": result["status"],
                "classified_artifacts": len(classified),
                "remaining_artifacts": len(unresolved),
            }
        )
    )


if __name__ == "__main__":
    main()
