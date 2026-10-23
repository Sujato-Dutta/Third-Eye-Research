"""Exercise final cohort preparation before reserving long GPU allocations."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

import finish_confirmation as driver
from test_admission_audit import data
from third_eye.config import Config
from third_eye.data.schema import Example
from third_eye.data.splits import create_manifest
from third_eye.experiments.confirmation import DECLARED
from third_eye.io import digest, file_digest, publish_jsonl_batch, write_json


def test_prepare_freezes_all_32_configs_disjoint_items_and_no_fitted_models(
    tmp_path, monkeypatch
):
    root = tmp_path / "work"
    root.mkdir()
    history = root / "history"
    history.mkdir()
    output = root / "cohort"
    output.mkdir()
    pilot = {"tasks": [], "manifest_audits": {}}
    final_by_task = {}
    for task in ("math", "code"):
        folder = root / task
        folder.mkdir()
        paths = {}
        for role, count in {
            "train": 1,
            "target_dev": 64,
            "ood_dev": 64,
            "retention_dev": 256,
        }.items():
            path = folder / (role + ".jsonl")
            publish_jsonl_batch(
                path,
                [
                    asdict(
                        Example(
                            f"{task}/{role}/{i}",
                            f"{task} {role} distinct prompt {i}",
                            "2",
                            role,
                            task,
                        )
                    )
                    for i in range(count)
                ],
            )
            paths[role] = path
        selection = folder / "selection.json"
        manifest = create_manifest(paths, selection)
        entries, final = {}, {}
        for role, count in {
            "target_test": 128,
            "ood_test": 136 if task == "code" else 128,
            "retention_test": 256,
        }.items():
            examples = [
                Example(
                    f"{task}/{role}/{i}",
                    f"{task} {role} distinct prompt {i}",
                    "2",
                    role,
                    task,
                )
                for i in range(count)
            ]
            path = folder / (role + ".jsonl")
            publish_jsonl_batch(path, [asdict(ex) for ex in examples])
            entries[role] = {
                "path": path.name,
                "sha256": file_digest(path),
                "count": count,
            }
            final[role] = examples
        final_path = folder / "final.json"
        write_json(
            final_path,
            {
                "schema_version": 1,
                "selection_hash": digest(manifest),
                "splits": entries,
            },
        )
        pilot["manifest_audits"][str(selection)] = {
            "sha256": file_digest(selection),
            "reserved_final_manifest": str(final_path),
            "reserved_final_manifest_sha256": file_digest(final_path),
        }
        final_by_task[task] = final
        for stream in (
            value for value in DECLARED["seed_grid"] if value.endswith("/" + task)
        ):
            cfg = Config()
            cfg = replace(
                cfg,
                protocol=replace(
                    cfg.protocol,
                    seed=12142,
                    candidate_size=2,
                    correction_attempts=8,
                    correction_seed_stride=16,
                    deterministic_execution=True,
                    amendment="A2",
                    candidate_sampling="stratified_distinct",
                    terminal_continuation=True,
                    status="frozen",
                ),
            )
            path = folder / (stream.split("/")[0] + ".json")
            write_json(path, cfg.to_dict())
            pilot["tasks"].append(
                {
                    "stream": stream,
                    "config": str(path),
                    "config_sha256": file_digest(path),
                    "manifest": str(selection),
                    "manifest_sha256": file_digest(selection),
                    "seed": 12142,
                }
            )
    pilot_path, pilot_report, classification = (
        root / "pilot.json",
        root / "pilot_report.json",
        root / "classification.json",
    )
    write_json(pilot_path, pilot)
    write_json(pilot_report, {"input_hashes": {}})
    write_json(
        classification,
        {"remaining_flagged_artifacts": [], "original_unreadable_artifacts": []},
    )
    code_audit = next(
        value
        for path, value in pilot["manifest_audits"].items()
        if "/code/" in path.replace("\\", "/")
    )
    ordered = sorted(
        final_by_task["code"]["ood_test"],
        key=lambda ex: digest(
            ["third-eye-cost-calibration-20261022", ex.id, ex.prompt_hash]
        ),
    )
    limits = {stream: 3600 for stream in DECLARED["seed_grid"]}
    counts = {
        task + "/" + role: DECLARED["scoring_items"][role.replace("_dev", "_test")]
        for task in ("math", "code")
        for role in ("target_dev", "ood_dev", "retention_dev")
    }
    design = {
        "inputs": {
            key: {"path": str(path), "sha256": file_digest(path)}
            for key, path in {
                "pilot_plan": pilot_path,
                "pilot_report": pilot_report,
                "classification_report": classification,
            }.items()
        },
        "rows": [
            {
                "parents": 32,
                "requested_target_ood_size": 128,
                "item_counts": counts,
                "cost": {
                    "base_plus_scoring_reservation_sus": 508,
                    "walltime_seconds_by_stream": limits,
                },
            }
        ],
        "calibration_reservation": {
            "purpose": "cost_only_excluded_from_confirmation_scoring",
            "original_final_manifest": code_audit["reserved_final_manifest"],
            "original_final_sha256": code_audit["reserved_final_manifest_sha256"],
            "items": [
                {"id": ex.id, "prompt_sha256": ex.prompt_hash} for ex in ordered[:8]
            ],
        },
    }
    for i, stream in enumerate(sorted(DECLARED["seed_grid"])):
        benchmark = root / f"benchmark_{i}.json"
        write_json(
            benchmark,
            {
                "task": {"stream": stream},
                "observations": [
                    {
                        "role": "ood_dev",
                        "generation_seconds": 16,
                        "verification_seconds": 0,
                        "count": 16,
                    }
                ],
            },
        )
        design["inputs"][f"benchmark_{i}"] = {
            "path": str(benchmark),
            "sha256": file_digest(benchmark),
        }
    design_path, receipt = root / "design.json", root / "design_complete.json"
    write_json(design_path, design)
    write_json(receipt, {"report_sha256": file_digest(design_path)})
    states = data()
    monkeypatch.setattr(
        driver, "load_pilot_states", lambda _: (states, [{"complete": True}] * 16, {})
    )
    rng = np.random.default_rng(42)
    point = np.array([-0.03, 0, 0, 0, 0, 0, 0.1])
    monkeypatch.setattr(
        driver,
        "bootstrap",
        lambda *a, **k: (point, point + rng.normal(0, 0.001, (2000, 7)), None, None),
    )
    request = {
        "source_tree_sha256": "fixture",
        "request_sha256": "request",
        "output": str(output),
        "inputs": {
            "design_report": {
                "path": str(design_path),
                "sha256": file_digest(design_path),
            },
            "design_completion": {"path": str(receipt), "sha256": file_digest(receipt)},
        },
        "history_roots": [str(history)],
        "walltime_seconds_by_stream": limits,
    }
    driver.prepare(request)
    plan = driver.checked_plan(request)
    assert len(plan["tasks"]) == len({task["seed"] for task in plan["tasks"]}) == 32
    assert plan["declaration"]["forecaster_fits"] == 0
    assert plan["power"]["conditional_planned_power"] > 0.99
    scoring_path = output / "scoring/code/ood_test.jsonl"
    selected = [
        json.loads(line)["id"] for line in scoring_path.read_text().splitlines()
    ]
    assert len(selected) == 128 and set(selected).isdisjoint(
        ex.id for ex in ordered[:8]
    )
    path = plan["tasks"][0]["config"]
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Frozen cohort file changed"):
        driver.checked_plan(request)
