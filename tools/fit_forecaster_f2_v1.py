"""One frozen fifteen-fit F2 study, CPU only; no test evaluation or scaling."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from third_eye.forecasting.dataset import read_records, split_records
from third_eye.io import file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.statistics.report import gate2
from extensions.forecasting_f2.training import train, ContrastForecaster

BASE = ROOT / "runs/a2"
SNAPSHOT = BASE / "provisional_forecasting_fast_20261008"
OUT = BASE / "forecaster_f2_v1"


def specifications():
    return [
        {"variant": variant, "horizon": horizon, "seed": seed}
        for variant in ["full", "independent"]
        for horizon in [2, 1]
        for seed in [42, 43, 44]
    ] + [
        {"variant": variant, "horizon": 2, "seed": 42}
        for variant in ["no_set", "no_temporal_aux", "no_comparative_loss"]
    ]


def name(task):
    return f"{task['variant']}_h{task['horizon']}_seed{task['seed']}"


def frozen():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("F2 validation and fitting require a Slurm compute node")
    plan = json.loads((BASE / "execution_plan.json").read_text())
    policy = plan["resource_policy"]
    assert source_inventory(ROOT)["source_tree_sha256"] == policy["source_tree_sha256"]
    assert all(
        file_digest(ROOT / p) == h
        for p, h in policy["operational_files_sha256"].items()
    )
    release = json.loads((BASE / "forecaster_f2_release_v1.json").read_text())
    assert all(file_digest(ROOT / p) == h for p, h in release["files"].items())
    assert (
        file_digest(SNAPSHOT / "snapshot_manifest.json")
        == release["snapshot_manifest_sha256"]
    )
    manifest = json.loads((SNAPSHOT / "snapshot_manifest.json").read_text())
    paths = []
    for item in manifest["trajectories"]:
        p = SNAPSHOT / "snapshot" / f"task_{item['index']}" / "meta_labels.jsonl"
        assert file_digest(p) == item["snapshot_labels_sha256"]
        paths.append(p)
    parts = split_records(read_records(paths), 42)
    assert {
        p: sorted({r["trajectory_id"] for r in rows}) for p, rows in parts.items()
    } == manifest["trajectory_assignments"]
    # Return only development partitions. Test rows are never passed to models.
    return release, {p: parts[p] for p in ["train", "validation"]}


def worker(index):
    release, parts = frozen()
    task = specifications()[index]
    f = train(parts["train"], parts["validation"], OUT / "models" / name(task), **task)
    restored = ContrastForecaster.load(OUT / "models" / name(task))
    import numpy as np

    assert np.array_equal(
        f.predict(parts["validation"]), restored.predict(parts["validation"])
    )
    write_json(
        OUT / "proofs" / f"{name(task)}.json",
        {
            "reload_exact": True,
            "release_snapshot_sha256": release["snapshot_manifest_sha256"],
            "weights_sha256": restored.metadata["weights_sha256"],
            "parameter_count": restored.metadata["trainable_parameters"],
            "test_evaluated": False,
        },
    )


def collect():
    comparisons = []
    for task in specifications():
        p = OUT / "models" / name(task)
        meta = json.loads((p / "metadata.json").read_text())
        metrics = json.loads((p / "metrics.json").read_text())
        decision = gate2(metrics["validation"], meta["kind"], meta["horizon"])
        comparisons.append(
            {
                **task,
                "name": name(task),
                "gate2": decision,
                "validation_future_h2": metrics["validation"]["future_h2"],
                "validation_head_state_rankings": metrics["validation"][
                    "head_state_rankings"
                ],
                "train_future_h2": metrics["train"]["future_h2"],
                "weights_sha256": meta["weights_sha256"],
                "epochs": meta["epochs_completed"],
                "best_epoch": meta["best_epoch"],
                "parameters": meta["trainable_parameters"],
                "seconds": meta["seconds"],
                "split_hashes": meta["split_hashes"],
            }
        )
    assert (
        len({json.dumps(c["split_hashes"], sort_keys=True) for c in comparisons}) == 1
    )
    for seed in [42, 43, 44]:
        a = next(c for c in comparisons if c["name"] == f"full_h2_seed{seed}")
        b = next(c for c in comparisons if c["name"] == f"full_h1_seed{seed}")
        assert a["parameters"] == b["parameters"]
    primary = next(c for c in comparisons if c["name"] == "full_h2_seed42")
    write_json(
        OUT / "review.json",
        {
            "status": "f2_forecaster_review_required",
            "primary_method": primary["name"],
            "primary_gate2": primary["gate2"],
            "comparisons": comparisons,
            "original_gate_miss_preserved": True,
            "test_evaluated": False,
            "online_submitted": False,
            "scaling_submitted": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "f2_forecaster_review_required",
                "primary_gate2": primary["gate2"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=range(15))
    args = parser.parse_args()
    if args.index is not None:
        worker(args.index)
        return
    release, parts = frozen()
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) >= 24
    assert not OUT.exists(), "Inspect immutable previous output before a new experiment"
    OUT.mkdir(parents=True)
    (OUT / "logs").mkdir()
    write_json(
        OUT / "predeclared_study.json",
        {
            "specifications": specifications(),
            "release": release,
            "states": {p: len({r["state_id"] for r in rs}) for p, rs in parts.items()},
            "test_evaluated": False,
            "selection": "Fixed full H2 seed42; no ablation/seed winner substitution",
        },
    )

    def run(index):
        with (OUT / "logs" / f"fit_{index:02d}.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(Path(__file__)), "--index", str(index)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return index

    workers = min(15, int(os.environ["SLURM_CPUS_PER_TASK"]) // 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, i) for i in range(15)]
        for future in as_completed(futures):
            print(f"Fixed F2 fit {future.result()} complete", flush=True)
    collect()


if __name__ == "__main__":
    main()
