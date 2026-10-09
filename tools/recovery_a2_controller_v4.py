"""Versioned timeout recovery; frozen A2 source and original artifacts stay intact."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from third_eye.config import Config
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory
from third_eye.cluster import require_gpu_allocation, scheduler_job_id
from third_eye.forecasting.dataset import read_records
from third_eye.data.schema import Correction, Example
from third_eye.data.splits import load_manifest
from third_eye.updates.candidates import sample_batches, batch_hash
from a2_after_labels import audit_labels

BASE = ROOT / "runs/a2"
REC = BASE / "recovery_20261008_v4"


def frozen():
    plan = json.loads((BASE / "execution_plan.json").read_text())
    policy = plan["resource_policy"]
    assert source_inventory(ROOT)["source_tree_sha256"] == policy["source_tree_sha256"]
    assert all(
        file_digest(ROOT / p) == h
        for p, h in policy["operational_files_sha256"].items()
    )
    release = json.loads((REC / "release_v4.json").read_text())
    assert all(file_digest(ROOT / p) == h for p, h in release["files"].items())
    return plan


def states():
    text = subprocess.check_output(
        ["sacct", "-j", "1049534", "-X", "-n", "-P", "--format=JobID,State,ExitCode"],
        text=True,
    )
    return {
        int(i): (state, exitcode)
        for i, state, exitcode in re.findall(
            r"^1049534_(\d+)\|([^|]+)\|([^\n]+)$", text, re.M
        )
    }


def output(task):
    cmd = task["commands"][0]
    return Path(cmd[cmd.index("--output") + 1])


def checkpoint(path, cfg):
    saved = json.loads((path / "state.json").read_text())
    assert saved["config_hash"] == cfg.fingerprint
    assert all(
        file_digest(path / name) == sha for name, sha in saved["weights"].items()
    )
    return saved


def validate_pool(pool):
    stats = pool["stats"]
    ids = [c["example"]["id"] for c in pool["items"]]
    assert (
        stats["harvest_complete"]
        and stats["correction_attempt_limit"] == 8
        and stats["correction_seed_stride"] == 16
    )
    assert len(ids) == len(set(ids)) == stats["verified_corrections"]
    assert all(1 <= c["attempt"] <= 8 for c in pool["items"])


def submit(script, arguments=(), dependencies=(), partition="gg", hours=1):
    argv = [
        "sbatch",
        "--parsable",
        "--account=IRI23021",
        f"--partition={partition}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={16 if partition == 'gh' else 4}",
        f"--time={hours}:00:00",
    ]
    if dependencies:
        argv.append("--dependency=" + ",".join(dependencies))
    argv.extend([str(ROOT / "tools" / script), *map(str, arguments)])
    from recovery_a2_relay_v4 import submit as relay_submit

    os.environ["THIRD_EYE_RECOVERY_QUEUE"] = str(REC / "submission_queue")
    return scheduler_job_id(relay_submit(argv[1:]))


def prepare(index):
    path = REC / f"task_{index}.json"
    if path.exists():
        assert (REC / f"submission_{index}.json").exists()
        return json.loads((REC / f"submission_{index}.json").read_text())["job_id"]
    assert states()[index][0] == "TIMEOUT", (
        "Only verified wall-time failures are automatically recovered"
    )
    plan = frozen()
    task = plan["stages"]["labels"][index]
    cmd = task["commands"][0]
    original = output(task)
    cfg = Config.load(cmd[cmd.index("--config") + 1])
    run = json.loads((original / "run.json").read_text())
    rows = read_records([original / "meta_labels.jsonl"])
    accepted = [
        json.loads(line)
        for line in (original / "ledger.jsonl").read_text().splitlines()
        if json.loads(line)["status"] == "accepted"
    ]
    generation = max(r["generation"] for r in accepted)
    assert generation == 4 and len(rows) == 12, (
        "Recovery expects four complete generations and one remaining"
    )
    audit_labels(original, cfg, {"completed_states": 4, "status": "running"})
    cp = original / "accepted" / f"generation_{generation}"
    saved = checkpoint(cp, cfg)
    meta = saved["metadata"]
    assert (
        meta["generation"] == generation
        and meta["trajectory_id"] == run["trajectory_id"]
        and meta["policy"] == "random"
        and meta["mode"] == "labels"
    )
    manifest, _ = load_manifest(cmd[cmd.index("--manifest") + 1])
    assert digest(manifest) == meta["manifest_hash"] == run["manifest_hash"]
    state_id = digest(
        {
            "parent": saved["adapter_hash"],
            "config": cfg.fingerprint,
            "splits": run["manifest_hash"],
            "generation": generation,
        }
    )[:20]
    partial = original / "states" / state_id
    pool = json.loads((partial / "correction_pool.json").read_text())
    validate_pool(pool)
    corrections = [
        Correction(Example(**c["example"]), c["completion"], c["attempt"])
        for c in pool["items"]
    ]
    seed = cfg.protocol.seed + generation * 1_000_000
    batches = sample_batches(corrections, 2, 3, seed, cfg.protocol.candidate_sampling)
    branches = []
    for k, batch in enumerate(batches):
        bh = batch_hash(batch)
        cid = f"k{k}-{bh[:10]}"
        branch = partial / cid
        assert json.loads((branch / "batch.json").read_text()) == [
            c.to_dict() for c in batch
        ]
        item = {
            "candidate_id": cid,
            "batch_hash": bh,
            "complete": (branch / "record.json").exists(),
        }
        if (branch / "t1/state.json").exists():
            item["t1_hash"] = checkpoint(branch / "t1", cfg)["adapter_hash"]
        if item["complete"]:
            row = json.loads((branch / "record.json").read_text())
            assert (
                row["seed"] == cfg.protocol.seed
                and row["model"] == cfg.model.name
                and row["protocol_amendment"] == "A2"
                and row["candidate_id"] == cid
            )
            assert (
                row["state_id"] == state_id
                and row["generation"] == generation
                and row["config_hash"] == cfg.fingerprint
            )
            assert (
                row["manifest_hash"] == run["manifest_hash"]
                and row["trajectory_id"] == run["trajectory_id"]
                and row["parent_adapter_hash"] == saved["adapter_hash"]
            )
            assert (
                row["batch_hash"] == bh
                and row["candidate_adapter_hash"] == item["t1_hash"]
            )
            assert (
                row["precommit"]["pool_statistics"] == pool["stats"]
                and row["precommit"]["history"] == meta["history"][-3:]
            )
            continuation = json.loads((branch / "continuation_pool.json").read_text())
            validate_pool(continuation)
            assert (
                row["continuation"]["pool_statistics"] == continuation["stats"]
                and row["continuation"]["seed"] == seed + 600_000
            )
            cbatch = json.loads((branch / "continuation_batch.json").read_text())
            ci = {c["example"]["id"]: c for c in continuation["items"]}
            available = int(len(continuation["items"]) >= 2)
            assert available == row["continuation_available"]
            assert row["runtime"]["candidate"]["optimizer_steps"] == 50 and row[
                "runtime"
            ]["continuation"]["optimizer_steps"] == (50 if available else 0)
            if available:
                assert (
                    len(cbatch) == 2 and len({c["example"]["id"] for c in cbatch}) == 2
                )
                assert all(c == ci.get(c["example"]["id"]) for c in cbatch)
                assert (
                    batch_hash(
                        [
                            Correction(
                                Example(**c["example"]), c["completion"], c["attempt"]
                            )
                            for c in cbatch
                        ]
                    )
                    == row["continuation"]["batch_hash"]
                )
            else:
                assert (
                    not cbatch
                    and row["candidate_adapter_hash"]
                    == row["continuation_adapter_hash"]
                    and row["evaluation"]["t1"] == row["evaluation"]["t2"]
                )
            assert (
                checkpoint(branch / "t2", cfg)["adapter_hash"]
                == row["continuation_adapter_hash"]
            )
            for head in ["target", "ood", "retention"]:
                assert all(
                    math.isfinite(row["evaluation"][when][head])
                    and 0 <= row["evaluation"][when][head] <= 1
                    for when in ["parent", "t1", "t2"]
                )
                for horizon, when in [("h1", "t1"), ("h2", "t2")]:
                    assert math.isclose(
                        row["labels"][horizon][head],
                        row["evaluation"][when][head]
                        - row["evaluation"]["parent"][head],
                        abs_tol=1e-12,
                    )
        branches.append(item)
    segment = REC / "segments" / f"task_{index}"
    context = {
        "index": index,
        "original": str(original),
        "segment": str(segment),
        "view": str(REC / "views" / f"task_{index}"),
        "checkpoint": str(cp),
        "generation": generation,
        "parent_adapter_hash": saved["adapter_hash"],
        "partial_state": str(partial),
        "branches": branches,
        "original_run_sha256": file_digest(original / "run.json"),
        "original_labels_sha256": file_digest(original / "meta_labels.jsonl"),
        "original_plan_sha256": file_digest(BASE / "execution_plan.json"),
    }
    input_paths = [
        original / "run.json",
        original / "meta_labels.jsonl",
        original / "ledger.jsonl",
        Path(cmd[cmd.index("--config") + 1]),
        cp / "state.json",
    ]
    input_paths.extend(
        p for p in partial.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}
    )
    context["input_sha256"] = {str(p): file_digest(p) for p in input_paths}
    argv = cmd[1:]
    argv[argv.index("--output") + 1] = str(segment)
    argv += ["--resume", str(cp), "--generations", "1"]
    context["argv"] = argv
    write_json(path, context)
    job = submit(
        "recovery_a2_gpu_v4.slurm",
        [index],
        [f"afterok:{os.environ['SLURM_JOB_ID']}"],
        partition="gh",
        hours=48,
    )
    write_json(
        REC / f"submission_{index}.json",
        {
            "job_id": job,
            "index": index,
            "manifest_sha256": file_digest(path),
            "original_failure": "TIMEOUT",
            "remaining_generations": 1,
            "cached_complete_branches": sum(b["complete"] for b in branches),
            "time_limit_hours": 48,
        },
    )
    print(
        json.dumps(
            {
                "recovery_index": index,
                "job_id": job,
                "cached_complete_branches": sum(b["complete"] for b in branches),
            }
        ),
        flush=True,
    )
    return job


def resume(index):
    require_gpu_allocation()
    frozen()
    path = REC / f"task_{index}.json"
    context = json.loads(path.read_text())
    receipt = json.loads((REC / f"submission_{index}.json").read_text())
    assert file_digest(path) == receipt["manifest_sha256"]
    assert all(file_digest(p) == sha for p, sha in context["input_sha256"].items())
    from third_eye.training import hf_backend
    import third_eye.experiments.labeling as labeling
    from recovery_a2_runtime_v4 import RecoveryRuntime

    real_backend, real_labeler = hf_backend.HFBackend, labeling.LabelGenerator
    runtimes = []

    def make_backend(cfg):
        runtime = RecoveryRuntime(
            real_backend(cfg), context, REC / "receipts" / f"task_{index}.jsonl"
        )
        runtimes.append(runtime)
        return runtime

    def make_labeler(*args, **kwargs):
        kwargs["feature_extractor"] = runtimes[0].extract_features
        return real_labeler(*args, **kwargs)

    hf_backend.HFBackend = make_backend
    labeling.LabelGenerator = make_labeler
    old_argv = sys.argv
    try:
        # Backend is created lazily by the frozen CLI; install cache hooks when it is constructed.
        def factory(cfg):
            runtime = make_backend(cfg)
            hooks = runtime.hooks()
            hooks.__enter__()
            runtime._recovery_hooks = hooks
            return runtime

        hf_backend.HFBackend = factory
        sys.argv = ["experiments/run.py", *context["argv"]]
        runpy.run_path(str(ROOT / "experiments/run.py"), run_name="__main__")
        audit_labels(
            Path(context["segment"]),
            runtimes[0].cfg,
            {"completed_states": 1, "status": "complete"},
        )
        write_json(
            REC / f"success_{index}.json",
            {
                "status": "complete",
                "original_trajectory_id": json.loads(
                    (Path(context["original"]) / "run.json").read_text()
                )["trajectory_id"],
                "manifest_sha256": file_digest(path),
                "job_id": os.environ["SLURM_JOB_ID"],
                "source_tree_sha256": source_inventory(ROOT)["source_tree_sha256"],
                "cache_receipts": str(REC / "receipts" / f"task_{index}.jsonl"),
                "logical_cached_timers": "Complete branches retain producer timers; partial T1 reuse records load time and unavailable producer timer; physical GPU cost includes original timeout and recovery allocations",
            },
        )
    finally:
        for runtime in runtimes:
            runtime._recovery_hooks.__exit__(None, None, None)
        hf_backend.HFBackend = real_backend
        labeling.LabelGenerator = real_labeler
        sys.argv = old_argv


def assemble(context, cfg):
    original, segment, view = [
        Path(context[k]) for k in ["original", "segment", "view"]
    ]
    assert (
        file_digest(original / "meta_labels.jsonl") == context["original_labels_sha256"]
    )
    rows = read_records([original / "meta_labels.jsonl", segment / "meta_labels.jsonl"])
    assert len(rows) == 15 and len({r["trajectory_id"] for r in rows}) == 1
    assert Counter(r["generation"] for r in rows) == {i: 3 for i in range(5)}
    assert len({r["state_id"] for r in rows}) == 5
    assert not view.exists()
    view.mkdir(parents=True)
    (view / "states").mkdir()
    prefix = {r["state_id"] for r in read_records([original / "meta_labels.jsonl"])}
    for sid in {r["state_id"] for r in rows}:
        source = (original if sid in prefix else segment) / "states" / sid
        (view / "states" / sid).symlink_to(source, target_is_directory=True)
    (view / "meta_labels.jsonl").write_bytes(
        (original / "meta_labels.jsonl").read_bytes()
        + (segment / "meta_labels.jsonl").read_bytes()
    )
    run = json.loads((original / "run.json").read_text())
    run["recovery_lineage"] = {
        "original": str(original),
        "segment": str(segment),
        "manifest_sha256": digest(context),
        "original_run_sha256": file_digest(original / "run.json"),
        "segment_run_sha256": file_digest(segment / "run.json"),
    }
    write_json(view / "run.json", run)
    write_json(
        view / "completed.json",
        {
            "status": "complete",
            "completed_states": 5,
            "requested_generations": 5,
            "recovered_from_timeout": True,
            "original": str(original),
            "segment": str(segment),
        },
    )
    audit_labels(view, cfg, {"completed_states": 5, "status": "complete"})
    return view


def barrier():
    plan = frozen()
    legacy = ["1057197", "1057269", "1057271", "1057272"]
    for job in legacy:
        state = subprocess.check_output(
            ["squeue", "-h", "-j", job, "-o", "%T"], text=True
        ).strip()
        if state != "PENDING":
            raise RuntimeError(
                "V3 job changed state; inspect before replacement: " + job + " " + state
            )
    status = states()
    assert len(status) == 36
    assert all(s in {"COMPLETED", "TIMEOUT"} for s, e in status.values())
    assert all(e == "0:0" for s, e in status.values() if s == "COMPLETED")
    recovered = [i for i, (s, _) in status.items() if s == "TIMEOUT"]
    jobs = [prepare(i) for i in recovered]
    derivative = deepcopy(plan)
    for i in recovered:
        cmd = derivative["stages"]["labels"][i]["commands"][0]
        cmd[cmd.index("--output") + 1] = str(REC / "views" / f"task_{i}")
    derivative["recovery_lineage"] = {
        "original_plan_sha256": file_digest(BASE / "execution_plan.json"),
        "timeout_indices": recovered,
        "recovery_jobs": jobs,
    }
    write_json(REC / "execution_plan.json", derivative)
    dependencies = [f"afterok:{os.environ['SLURM_JOB_ID']}"]
    for job in jobs:
        text = (
            subprocess.check_output(
                ["sacct", "-j", job, "-X", "-n", "-P", "--format=State,ExitCode"],
                text=True,
            )
            .strip()
            .splitlines()
        )
        assert len(text) == 1
        if text[0] == "COMPLETED|0:0":
            continue
        assert text[0].split("|")[0] in {"PENDING", "RUNNING", "COMPLETING"}
        dependencies.append(f"afterok:{job}")
    finishjob = submit("recovery_a2_cpu_v4.slurm", ["finish"], dependencies, hours=4)
    write_json(
        REC / "handoff.json",
        {
            "final_cpu_job": finishjob,
            "recovery_gpu_jobs": jobs,
            "timeout_indices": recovered,
            "derivative_plan_sha256": file_digest(REC / "execution_plan.json"),
            "original_forecaster_job": 1049535,
        },
    )
    for job in legacy:
        state = subprocess.check_output(
            ["squeue", "-h", "-j", job, "-o", "%T"], text=True
        ).strip()
        if state != "PENDING":
            # Preserve running V3 work; retire only the still-pending duplicate V4 graph.
            for replacement in [*jobs, finishjob]:
                if (
                    subprocess.check_output(
                        ["squeue", "-h", "-j", replacement, "-o", "%T"], text=True
                    ).strip()
                    == "PENDING"
                ):
                    subprocess.run(["scancel", replacement], check=True)
            raise RuntimeError(
                "V3 changed state during handoff; review required: " + job
            )
    subprocess.run(["scancel", *legacy], check=True)
    write_json(
        REC / "retired_v3_queue.json",
        {
            "jobs": legacy,
            "reason": "Validated V4 reuses exact retained partial M1; original source and evidence preserved",
            "replacement_gpu_jobs": jobs,
            "replacement_cpu_job": finishjob,
        },
    )
    old = subprocess.check_output(
        ["squeue", "-h", "-j", "1049535", "-o", "%T"], text=True
    ).strip()
    if old == "PENDING":
        subprocess.run(["scancel", "1049535"], check=True)
    print(
        json.dumps({"replacement_cpu_job": finishjob, "recovery_gpu_jobs": jobs}),
        flush=True,
    )


def finish():
    plan = frozen()
    derivative = json.loads((REC / "execution_plan.json").read_text())
    for i in derivative["recovery_lineage"]["timeout_indices"]:
        assert (
            json.loads((REC / f"success_{i}.json").read_text())["status"] == "complete"
        )
        cmd = plan["stages"]["labels"][i]["commands"][0]
        assemble(
            json.loads((REC / f"task_{i}.json").read_text()),
            Config.load(cmd[cmd.index("--config") + 1]),
        )
    os.environ["THIRD_EYE_RECOVERY_PLAN"] = str(REC / "execution_plan.json")
    runpy.run_path(str(ROOT / "tools/recovery_a2_after_labels.py"), run_name="__main__")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "resume", "barrier", "finish"])
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Recovery execution requires Slurm compute allocation")
    if a.mode in {"prepare", "resume", "barrier"}:
        passed = json.loads((REC / "cuda_passed.json").read_text())
        if passed["status"] != "passed" or passed["release_sha256"] != file_digest(
            REC / "release_v4.json"
        ):
            raise RuntimeError("V4 requires its exact three-checkpoint CUDA validation")
    if a.mode == "prepare":
        prepare(a.index)
    elif a.mode == "resume":
        resume(a.index)
    elif a.mode == "barrier":
        barrier()
    else:
        finish()


if __name__ == "__main__":
    main()
