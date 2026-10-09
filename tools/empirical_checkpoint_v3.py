# ruff: noqa: E402
"""E2: retained-checkpoint replication, fixed controls and automatic review stop."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import runpy
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_common_v1 import BASE, checked, prepared
from empirical_replication_v1 import methods
from third_eye.cluster import require_gpu_allocation, scheduler_job_id
from third_eye.config import Config
from third_eye.io import file_digest, write_json
from empirical_relay_v1 import atomic_json

OUT = BASE / "checkpoint_fallback_v3"


def frozen():
    checked()
    release = json.loads((OUT / "release.json").read_text())
    for name, sha in release["files"].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError(f"E2 source changed: {name}")
    for path, sha in release["prerequisites"].items():
        if file_digest(path) != sha:
            raise RuntimeError(f"E2 prerequisite changed: {path}")
    methods()
    decision = json.loads((OUT / "decision.json").read_text())
    if decision["decision"] != "adopt_checkpoint_based_balanced_fallback":
        raise RuntimeError("Recorded E2 adoption required")
    return release["scope"]


def passed(kind):
    receipt = json.loads((OUT / f"{kind}_passed.json").read_text())
    if receipt["status"] != "passed" or receipt["release_sha256"] != file_digest(
        OUT / "release.json"
    ):
        raise RuntimeError(f"Current E2 {kind} validation required")


def arm_deadline(scope):
    remaining = (
        datetime.fromisoformat(scope["deadline_utc"]) - datetime.now(timezone.utc)
    ).total_seconds()
    if remaining <= 0:
        raise RuntimeError("Experiment deadline passed")

    def expired(signum, frame):
        raise RuntimeError(
            "Predeclared experiment deadline reached; preserve incomplete evidence"
        )

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(remaining)))


def generation_for(seed):
    if seed not in (6142, 6143):
        raise ValueError("Only the frozen two-seed cohort is eligible")
    return 0 if seed == 6142 else 4


def arguments(mode, index=None, dependency=None):
    gpu = mode in {"trajectory", "noise"}
    args = [
        "--parsable",
        "--account=IRI23021",
        "--nodes=1",
        "--ntasks=1",
        "--partition=" + ("gh" if gpu else "gg"),
        "--cpus-per-task=" + ("16" if gpu else "72"),
        "--time="
        + ({"trajectory": "48:00:00", "noise": "24:00:00"}.get(mode, "1:00:00")),
    ]
    if dependency:
        args.append("--dependency=afterany:" + ":".join(map(str, dependency)))
    args += [
        str(ROOT / f"tools/empirical_checkpoint_{'gpu' if gpu else 'cpu'}_v2.slurm"),
        mode,
    ]
    if index is not None:
        args.append(str(index))
    return args


def publish_noise_input(task, rows):
    """Publish a complete immutable state, never hash an actively appended ledger."""
    generation = generation_for(task["seed"])
    if len(rows) != 3 or {r["generation"] for r in rows} != {generation}:
        raise RuntimeError("Expected exactly the predeclared K=3 parent state")
    if (
        len({r["state_id"] for r in rows}) != 1
        or len({r["batch_hash"] for r in rows}) != 3
    ):
        raise RuntimeError("State or composition mismatch")
    state = Path(task["output"]) / "states" / rows[0]["state_id"]
    files = sorted(p for p in state.rglob("*") if p.is_file())
    # Require actual saved weights at all seven points; no reconstruction fallback.
    for folder in [
        state / "parent",
        *[Path(r["checkpoints"][h]) for r in rows for h in ("t1", "t2")],
    ]:
        if (
            not folder.is_relative_to(state)
            or not (folder / "state.json").is_file()
            or not list(folder.glob("*.safetensors"))
        ):
            raise RuntimeError("Retained checkpoint missing or outside selected state")
    record = dict(
        task,
        source=task["output"],
        generation=generation,
        selected_records=rows,
        state_id=rows[0]["state_id"],
        trajectory_id=rows[0]["trajectory_id"],
        input_sha256={str(p): file_digest(p) for p in files},
        release_sha256=file_digest(OUT / "release.json"),
    )
    target = OUT / "noise_inputs" / f"task_{task['index']}.json"
    if target.exists():
        raise RuntimeError(
            "Noise state already published; inspect instead of replacing"
        )
    atomic_json(target, record)
    return record


def load_noise_input(index):
    task = json.loads((OUT / "noise_inputs" / f"task_{index}.json").read_text())
    expected = prepared()["replication_tasks"][index]
    if any(task[k] != expected[k] for k in expected) or task[
        "generation"
    ] != generation_for(expected["seed"]):
        raise RuntimeError("Noise sample no longer equals predeclared cohort")
    if task["release_sha256"] != file_digest(OUT / "release.json") or any(
        file_digest(p) != sha for p, sha in task["input_sha256"].items()
    ):
        raise RuntimeError("Noise state files changed")
    return task


def run(index):
    scope = frozen()
    passed("cpu")
    passed("cuda")
    require_gpu_allocation()
    arm_deadline(scope)
    task = prepared()["replication_tasks"][index]
    if Path(task["output"]).exists():
        raise RuntimeError(
            "Never overwrite or silently restart an attempted trajectory"
        )
    from empirical_checkpoint_relay_v3 import submit
    import third_eye.experiments.labeling as labeling

    original = labeling.LabelGenerator
    os.environ["THIRD_EYE_EMPIRICAL_QUEUE"] = str(OUT / "submission_queue")

    class CheckpointLabeler(original):
        def label_state(self, generation, history=()):
            rows = super().label_state(generation, history)
            if generation == generation_for(task["seed"]):
                publish_noise_input(task, rows)
                receipt = OUT / "noise_submissions" / f"task_{index}.json"
                atomic_json(receipt, dict(status="submitting", index=index))
                job = scheduler_job_id(submit(arguments("noise", index)))
                atomic_json(
                    receipt,
                    dict(
                        status="queued",
                        index=index,
                        job_id=job,
                        release_sha256=file_digest(OUT / "release.json"),
                    ),
                )
            return rows

    labeling.LabelGenerator = CheckpointLabeler
    sys.argv = [
        str(ROOT / "experiments/run.py"),
        "--config",
        task["config"],
        "--manifest",
        task["manifest"],
        "--output",
        task["output"],
        "--mode",
        "labels",
        "--policy",
        "random",
        "--generations",
        "5",
        "--keep-accepted",
        "5",
    ]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        labeling.LabelGenerator = original


def launch():
    frozen()
    passed("cpu")
    passed("cuda")
    from empirical_checkpoint_relay_v3 import submit

    os.environ["THIRD_EYE_EMPIRICAL_QUEUE"] = str(OUT / "submission_queue")
    target = OUT / "trajectory_submissions.json"
    if target.exists():
        raise RuntimeError("Launch already attempted; inspect receipts before recovery")
    record = dict(
        status="submitting", jobs=[], release_sha256=file_digest(OUT / "release.json")
    )
    atomic_json(target, record)
    for task in prepared()["replication_tasks"]:
        if Path(task["output"]).exists():
            raise RuntimeError("Fresh cohort must have no earlier outcomes")
        job = scheduler_job_id(submit(arguments("trajectory", task["index"])))
        record["jobs"].append(
            dict(
                index=task["index"],
                stream=task["stream"],
                seed=task["seed"],
                job_id=job,
            )
        )
        atomic_json(target, record)
    record["confirmation_job"] = scheduler_job_id(
        submit(arguments("confirm", dependency=[j["job_id"] for j in record["jobs"]]))
    )
    record["status"] = "queued_with_checkpoint_noise_and_review_stop"
    atomic_json(target, record)
    print(json.dumps(record), flush=True)


def cpu_validate():
    frozen()
    import shutil

    if shutil.disk_usage(OUT).free < 40 * 1024**3:
        raise RuntimeError(
            "Insufficient scratch space to retain checkpoints and evidence"
        )
    if (
        shutil.disk_usage("/work/11617/sujato_ts/vista/third_eye_results").free
        < 40 * 1024**3
    ):
        raise RuntimeError("Insufficient durable archive space")
    tasks = prepared()["replication_tasks"]
    assert len(tasks) == 8 and len({(t["stream"], t["seed"]) for t in tasks}) == 8
    assert {t["seed"] for t in tasks} == {6142, 6143}
    assert all(t["generations"] == 5 and not Path(t["output"]).exists() for t in tasks)
    for folder in ("noise_inputs", "noise_submissions", "noise", "verification"):
        (OUT / folder).mkdir(exist_ok=True)
    write_json(
        OUT / "cpu_passed.json",
        dict(status="passed", release_sha256=file_digest(OUT / "release.json")),
    )


def cuda_validate():
    frozen()
    passed("cpu")
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.benchmarks import make_verifier
    from empirical_noise_v1 import batch, same_hash
    from empirical_checkpoint_noise_v3 import collect_items
    import torch

    results = []
    seen = set()
    for task in prepared()["noise_tasks"]:
        if task["stream"] in seen:
            continue
        seen.add(task["stream"])
        checkpoint = sorted((Path(task["source"]) / "accepted").glob("generation_*"))[
            -1
        ]
        state = json.loads((checkpoint / "state.json").read_text())
        cfg = Config.load(task["config"])
        backend = HFBackend(cfg)
        backend.load_checkpoint(checkpoint)
        same_hash(backend, state["adapter_hash"], "retained operational checkpoint")
        parent = backend.snapshot()
        _, splits = load_manifest(task["manifest"])
        verifier = make_verifier()
        items = batch(task, task["selected_records"][0])
        log = backend.train(
            items,
            cfg.protocol.seed + 50_000,
            log_path=OUT / "verification" / f"{len(results)}_training.jsonl",
        )
        assert log["optimizer_steps"] == 50
        backend.restore(parent)
        same_hash(backend, state["adapter_hash"], "operational training rollback")
        tiny = {
            role: examples[:2]
            for role, examples in splits.items()
            if role.endswith("_dev")
        }
        measured = collect_items(
            backend, tiny, verifier, cfg.protocol, cfg.protocol.seed
        )
        # A terminal continuation performs zero training and preserves the loaded state.
        terminal = dict(
            measured,
            continuation_available=0,
            training=dict(optimizer_steps=0, reserved_optimizer_steps=50),
        )
        assert (
            terminal["items"] == measured["items"]
            and terminal["adapter_hash"] == state["adapter_hash"]
        )
        results.append(
            dict(
                stream=task["stream"],
                adapter_hash=state["adapter_hash"],
                optimizer_steps=50,
                rollback_exact=True,
                terminal_preserved=True,
                role_counts={r: len(v) for r, v in measured["items"].items()},
            )
        )
        del backend, parent
        gc.collect()
        torch.cuda.empty_cache()
    assert len(results) == 4
    write_json(
        OUT / "cuda_passed.json",
        dict(
            status="passed",
            streams=results,
            release_sha256=file_digest(OUT / "release.json"),
        ),
    )


def confirm():
    frozen()
    from empirical_checkpoint_relay_v3 import submit
    from empirical_replication_v1 import confirm as replication_confirm

    os.environ["THIRD_EYE_EMPIRICAL_QUEUE"] = str(OUT / "submission_queue")
    # Queue the reliability review even if a native trajectory failed; it records coverage.
    receipts = [OUT / "noise_submissions" / f"task_{i}.json" for i in range(8)]
    jobs = [
        json.loads(p.read_text())["job_id"]
        for p in receipts
        if p.exists() and json.loads(p.read_text()).get("status") == "queued"
    ]
    job = scheduler_job_id(submit(arguments("review", dependency=jobs or None)))
    atomic_json(
        OUT / "review_submission.json",
        dict(job_id=job, dependency_jobs=jobs, missing_noise_submissions=8 - len(jobs)),
    )
    replication_confirm()


def review():
    frozen()
    from empirical_noise_review_v1 import inspect, aggregate

    records, missing = [], []
    for index in range(8):
        path = OUT / "noise" / f"task_{index}/completed.json"
        if not path.exists():
            missing.append(index)
            continue
        record = json.loads(path.read_text())
        if record["status"] != "complete" or record[
            "noise_release_sha256"
        ] != file_digest(OUT / "release.json"):
            raise RuntimeError("Invalid E2 noise release")
        load_noise_input(index)
        measured = inspect(record)
        task = load_noise_input(index)
        from empirical_noise_review_v1 import reversal, regret

        one = [r["labels"]["utility_h1"] for r in task["selected_records"]]
        two = [r["labels"]["utility_h2"] for r in task["selected_records"]]
        measured["native_recorded_h1_h2_reversal"] = reversal(one, two)
        measured["native_recorded_greedy_regret"] = regret(one, two)
        measured["native_vs_reevaluated_metrics"] = {
            c["candidate_id"]: {
                n: c["conditions"][n]["reevaluation_minus_original"]
                for n in ("original_m1", "original_m2")
            }
            for c in record["candidates"]
        }
        measured["fixed_prediction_verifier_repeats"] = {
            c["candidate_id"]: {
                n: v.get("fixed_prediction_verifier_repeats", {})
                for n, v in c["conditions"].items()
            }
            for c in record["candidates"]
        }
        records.append(measured)
    target = OUT / "noise_review.json"
    if target.exists():
        raise RuntimeError("Review output already exists")
    write_json(
        target,
        dict(
            status="incomplete_review_stop" if missing else "science_review_required",
            complete_states=len(records),
            missing_indices=missing,
            all=aggregate(records) if records else {},
            streams={
                s: aggregate([r for r in records if r["stream"] == s])
                for s in sorted({r["stream"] for r in records})
            },
            parent_results=records,
            original_gate2_unchanged=True,
            online_or_scaling_submitted=False,
            reliability_sample_independent_trajectories=8,
            interpretation="Matched repeatability evidence; no post-hoc scientific pass threshold; eight units limit precision",
        ),
    )
    archive()
    if missing:
        raise RuntimeError(
            f"Incomplete E2 reliability cohort: {missing}; no replacement"
        )


def archive():
    import tarfile

    durable = Path("/work/11617/sujato_ts/vista/third_eye_results")
    target = durable / "empirical_checkpoint_fallback_v3.tar.gz"
    if target.exists():
        raise RuntimeError("Durable archive already exists")
    partial = target.with_suffix(target.suffix + ".partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:
        for path in (OUT, BASE / "replication", BASE / "replication_confirmation"):
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(ROOT)))
        release = json.loads((OUT / "release.json").read_text())
        for name in release["files"]:
            tar.add(ROOT / name, arcname=name)
    partial.replace(target)
    write_json(
        OUT / "durable_archive.json",
        dict(path=str(target), sha256=file_digest(target), bytes=target.stat().st_size),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "validate",
            "verify",
            "launch",
            "trajectory",
            "noise",
            "confirm",
            "review",
        ],
    )
    parser.add_argument("index", type=int, nargs="?", choices=range(8))
    args = parser.parse_args()
    if args.mode in {"trajectory", "noise"} and args.index is None:
        parser.error("Fixed task index required")
    actions = {
        "validate": cpu_validate,
        "verify": cuda_validate,
        "launch": launch,
        "confirm": confirm,
        "review": review,
    }
    if args.mode == "trajectory":
        run(args.index)
    elif args.mode == "noise":
        from empirical_checkpoint_noise_v3 import measure

        measure(args.index)
    else:
        actions[args.mode]()


if __name__ == "__main__":
    main()
