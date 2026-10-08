"""One predeclared, matched forecaster diagnostic; original results stay intact."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from forecast_a2_prefix_20261008 import assignments, remap
from third_eye.forecasting.dataset import read_records
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.statistics.report import gate2

BASE = ROOT / "runs/a2"
SNAPSHOT = BASE / "provisional_forecasting_fast_20261008"
OUT = BASE / "forecaster_patience_diagnostic_20261008"


def diagnostic_command(command, paths):
    result = remap(command, paths, BASE / "study", OUT / "study")
    for flag, value in [("--epochs", "200"), ("--patience", "200")]:
        if flag in result:
            result[result.index(flag) + 1] = value
        else:
            result += [flag, value]
    assert result[result.index("--device") + 1] == "cpu"
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Diagnostic requires scheduled compute allocation")
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) >= 24
    plan = json.loads((BASE / "execution_plan.json").read_text())
    assert (
        source_inventory(ROOT)["source_tree_sha256"]
        == plan["resource_policy"]["source_tree_sha256"]
    )
    assert all(
        file_digest(ROOT / p) == h
        for p, h in plan["resource_policy"]["operational_files_sha256"].items()
    )
    release = json.loads(
        (BASE / "patience_diagnostic_release_20261008.json").read_text()
    )
    assert all(file_digest(ROOT / p) == h for p, h in release["files"].items())
    assert (
        file_digest(SNAPSHOT / "snapshot_manifest.json")
        == release["snapshot_manifest_sha256"]
    )
    manifest = json.loads((SNAPSHOT / "snapshot_manifest.json").read_text())
    paths = []
    for item in manifest["trajectories"]:
        path = SNAPSHOT / "snapshot" / f"task_{item['index']}" / "meta_labels.jsonl"
        assert file_digest(path) == item["snapshot_labels_sha256"]
        paths.append(str(path))
    rows = read_records(paths)
    assert assignments(rows) == manifest["trajectory_assignments"]
    assert not OUT.exists(), "Diagnostic outputs are immutable"
    OUT.mkdir(parents=True)
    tasks = []
    for task in plan["stages"]["forecasters"]:
        cmd = task["commands"][0]
        if (
            cmd[cmd.index("--kind") + 1] in {"direct", "matched_h1"}
            and "--ablate" not in cmd
            and "--scalar" not in cmd
        ):
            tasks.append(diagnostic_command(cmd, paths))
    identities = {
        (c[c.index("--kind") + 1], int(c[c.index("--seed") + 1])) for c in tasks
    }
    assert identities == {
        (kind, seed) for kind in ["direct", "matched_h1"] for seed in [42, 43, 44]
    }
    assert len(tasks) == 6
    write_json(
        OUT / "preregistered_execution.json",
        {
            "diagnostic_only": True,
            "fixed_primary_seed": 42,
            "snapshot_manifest_sha256": release["snapshot_manifest_sha256"],
            "change": "Patience 20 to 200; existing maximum 200 epochs unchanged; checkpoint selection unchanged",
            "fixed_commands": tasks,
            "test_metrics_used_for_selection": False,
            "candidate_and_h2_protocol_changed": False,
            "gate_thresholds_changed": False,
        },
    )

    def fit(i, cmd):
        with (OUT / f"fit_{i}.log").open("w") as log:
            subprocess.run(
                [sys.executable, *cmd], stdout=log, stderr=subprocess.STDOUT, check=True
            )
        return i

    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed(
            [pool.submit(fit, i, cmd) for i, cmd in enumerate(tasks)]
        ):
            print(f"Fixed diagnostic fit {future.result()} complete", flush=True)
    comparisons = []
    for seed in [42, 43, 44]:
        entries = {}
        for kind in ["direct", "matched_h1"]:
            p = OUT / "study/forecasters" / f"{kind}_seed{seed}"
            meta = json.loads((p / "metadata.json").read_text())
            metrics = json.loads((p / "metrics.json").read_text())["validation"]
            entries[kind] = {"metadata": meta, "validation": metrics}
        assert (
            entries["direct"]["metadata"]["split_hashes"]
            == entries["matched_h1"]["metadata"]["split_hashes"]
        )
        comparisons.append(
            {
                "seed": seed,
                "h2_gate": gate2(entries["direct"]["validation"], "direct", 2),
                "matched_h1_future_h2": entries["matched_h1"]["validation"][
                    "future_h2"
                ],
                "fit_epochs": {
                    k: e["metadata"]["epochs_completed"] for k, e in entries.items()
                },
                "weight_hashes": {
                    k: e["metadata"]["weights_sha256"] for k, e in entries.items()
                },
            }
        )
    write_json(
        OUT / "review.json",
        {
            "status": "forecaster_diagnostic_review_required",
            "diagnostic_only": True,
            "reference_seed": 42,
            "comparisons": comparisons,
            "original_snapshot_gate_passed": False,
            "online_submitted": False,
            "scaling_submitted": False,
            "test_metrics_used_for_selection": False,
        },
    )
    print(
        json.dumps(
            {
                "diagnostic_complete": True,
                "reference_seed_gate": comparisons[0]["h2_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
