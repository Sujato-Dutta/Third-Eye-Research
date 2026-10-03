"""Advance a scheduled core campaign using measured pilots and research gates."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.io import append_jsonl, write_json
from third_eye.cluster import scheduler_job_id
from third_eye.storage import check_storage
from plan_study import build_plan


def decision(path, number):
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if evidence.get("gate") != number or evidence.get("passed") is not True:
        raise RuntimeError(f"Gate {number} did not pass: {json.dumps(evidence)}")
    return evidence


def accounting(job):
    output = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            str(job),
            "--format=JobID,State,ElapsedRaw,ExitCode",
        ],
        text=True,
    )
    return [line.split("|")[:4] for line in output.splitlines() if line.strip()]


def wait_success(job, event):
    event("waiting", job_id=str(job))
    missing = 0
    while True:
        query = subprocess.run(
            ["squeue", "-h", "-j", str(job), "-o", "%i %T"],
            capture_output=True,
            text=True,
        )
        if query.returncode and "Invalid job id" not in query.stderr:
            raise RuntimeError("Scheduler query failed: " + query.stderr.strip())
        if not query.stdout.strip():
            records = accounting(job)
            if records:
                if any(row[1] != "COMPLETED" or row[3] != "0:0" for row in records):
                    raise RuntimeError(f"Job {job} failed: {records}")
                return sum(int(row[2]) for row in records) / 3600
            missing += 1
            if missing >= 10:
                raise RuntimeError(
                    f"Job {job} is missing from scheduler and accounting"
                )
        time.sleep(120)


def wall_limit(hours):
    minutes = max(15, math.ceil(hours * 60))
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def check_deadline(campaign, task_count, parallelism, estimate):
    if campaign.get("completion_deadline_utc"):
        deadline = datetime.fromisoformat(campaign["completion_deadline_utc"])
        if deadline.tzinfo is None:
            raise ValueError("Completion deadline must include its UTC offset")
        required = math.ceil(task_count / parallelism) * estimate * 3600
        if time.time() + required > deadline.timestamp():
            raise RuntimeError(
                "Estimated stage runtime exceeds the declared completion deadline"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cluster", choices=("dgx", "vista"), default="dgx")
    p.add_argument("--account")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--storage-quota-gb", type=float, default=50)
    p.add_argument("--gres", default="gpu:a100_4g.20gb:1")
    p.add_argument("--small-gres", default="gpu:a100_1g.5gb:1")
    p.add_argument("--budget-hours", type=float, default=160)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        p.error("Campaign continuation requires a scheduled compute node")
    if a.budget_hours <= 0 or a.storage_quota_gb <= 0 or a.max_parallel < 1:
        p.error("Budgets and parallelism must be positive")
    root = Path(a.output)
    if root.exists():
        p.error("Campaign output must be new")
    root.mkdir(parents=True)

    def event(status, **details):
        record = {"status": status, "timestamp": time.time(), **details}
        append_jsonl(root / "events.jsonl", record)
        write_json(root / "status.json", record)
        print(json.dumps(record), flush=True)

    def run(*argv):
        subprocess.run([sys.executable, *map(str, argv)], check=True)

    def backup():
        if a.cluster != "vista":
            return
        destination = Path(os.environ["WORK"]) / "third_eye_results" / root.name
        destination.mkdir(parents=True, exist_ok=True)
        # Only project sources, fixed data, proofs and result artifacts; model
        # caches and local credentials are not part of durable staging.
        selected = [
            root,
            Path("src"),
            Path("experiments"),
            Path("data/processed/v1"),
            Path("models/mirror_verification"),
            Path("runs/pilot"),
        ]
        selected.extend(Path.cwd().glob("requirements*.txt"))
        for path in selected:
            if not path.exists():
                continue
            relative = path.resolve().relative_to(Path.cwd())
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                command = ["rsync", "-a", str(path) + "/", str(target) + "/"]
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                command = ["rsync", "-a", str(path), str(target)]
            subprocess.run(command, check=True)

    try:
        campaign = json.loads(Path(a.campaign).read_text(encoding="utf-8"))
        pilots = campaign["pilots"]
        if len(pilots) != 4 or {(x["model"], x["task"]) for x in pilots} != {
            (model, task)
            for model in ("Qwen/Qwen3-4B", "meta-llama/Llama-3.2-3B-Instruct")
            for task in ("math", "code")
        }:
            raise ValueError("Core continuation requires all four model/task pilots")
        pilot_hours = campaign.get("prior_gpu_hours", 0)
        if campaign.get("pending_verification_job"):
            pilot_hours += wait_success(campaign["pending_verification_job"], event)
        for pilot in pilots:
            pilot_hours += wait_success(pilot["job_id"], event)
            completed = json.loads((Path(pilot["run"]) / "completed.json").read_text())
            if completed.get("status") != "complete" or not completed.get("accepted"):
                raise RuntimeError("A pilot did not complete a matched H=2 state")
        configs = []
        state_hours = {}
        for pilot in pilots:
            directory = Path(pilot["run"])
            metadata = json.loads((directory / "run.json").read_text())
            ledger = [
                json.loads(line)
                for line in (directory / "ledger.jsonl").read_text().splitlines()
            ]
            states = [
                row["seconds"] / 3600
                for row in ledger
                if row.get("status") == "complete"
            ]
            if not states:
                raise RuntimeError("Missing complete-state runtime measurements")
            state_hours[(pilot["model"], pilot["task"])] = max(states)
            if pilot["task"] == "math":
                output = (
                    root / "frozen" / (pilot["model"].split("/")[-1].lower() + ".json")
                )
                run(
                    "experiments/freeze_protocol.py",
                    "--config",
                    pilot["config"],
                    "--pilot-run",
                    directory,
                    "--depth",
                    "5",
                    "--output",
                    output,
                )
                configs.append(output)
            partner = next(
                x
                for x in pilots
                if x["model"] == pilot["model"] and x["task"] != pilot["task"]
            )
            other = json.loads((Path(partner["run"]) / "run.json").read_text())
            if metadata["config_hash"] != other["config_hash"] or metadata.get(
                "model_source"
            ) != other.get("model_source"):
                raise ValueError(
                    "Math/code pilots must measure the same fixed protocol and model source"
                )
        budget = a.budget_hours - pilot_hours
        if budget <= 0:
            raise RuntimeError("Pilots exhausted the GPU-hour budget")
        run_root = root / "study"
        plan_path = build_plan(configs, campaign["data_root"], root / "plan", run_root)
        plan = json.loads(plan_path.read_text())
        plan["storage_quota_gb"] = a.storage_quota_gb
        plan["cluster"] = a.cluster
        if a.cluster == "vista":
            plan.pop("compute_budget_a100_hours", None)
            plan["compute_budget_h200_node_hours"] = budget
        core_plan_path, core_plan = plan_path, plan
        gate1_path = run_root / "gate1_initial" / "gate1.json"
        for stage in ("forecasters", "online_core", "final_core"):
            for task in plan["stages"][stage]:
                task.setdefault("requires_gates", []).append(str(gate1_path))
        write_json(plan_path, plan)
        # The first trajectory in each model/task group yields 20 complete
        # states at T=5. Selection is predetermined and independent of labels.
        initial = [0, 9, 18, 27]
        label_files = [
            task["commands"][0][task["commands"][0].index("--output") + 1]
            + "/meta_labels.jsonl"
            for task in plan["stages"]["labels"]
        ]
        largest_state = max(state_hours.values())
        label_estimate = 5 * largest_state * 1.25
        if label_estimate > 12:
            raise RuntimeError(
                "Measured label trajectories exceed the 12-hour task ceiling; uniformly repilot"
            )
        event(
            "planned",
            plan=str(plan_path),
            pilot_gpu_hours=pilot_hours,
            remaining_gpu_hours=budget,
            label_task_estimate_hours=label_estimate,
            state_hours={
                f"{model}:{task}": hours for (model, task), hours in state_hours.items()
            },
            deferred="Transfer and stress require measured core evidence and remaining budget",
        )

        def submit(stage, indices=None, estimate=None, small=False):
            estimate = label_estimate if estimate is None else estimate
            check_deadline(
                campaign,
                len(plan["stages"][stage]) if indices is None else len(indices),
                a.max_parallel,
                estimate,
            )
            argv = [
                sys.executable,
                "experiments/submit_stage.py",
                "--plan",
                str(plan_path),
                "--stage",
                stage,
                "--cluster",
                a.cluster,
                "--max-parallel",
                str(a.max_parallel),
                "--estimated-task-hours",
                str(estimate),
                "--budget-hours",
                str(budget),
                "--time-limit",
                wall_limit(estimate),
                "--submit",
            ]
            if a.cluster == "dgx":
                argv.extend(("--gres", a.small_gres if small else a.gres))
            if a.account:
                argv.extend(("--account", a.account))
            if indices is not None:
                argv.extend(("--indices", *map(str, indices)))
            result = subprocess.run(argv, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            print(result.stdout, flush=True)
            match = re.search(r"Submitted job (\d+)", result.stdout)
            if not match:
                raise RuntimeError("Scheduler submission did not return a job ID")
            job = match.group(1)
            event(
                "submitted",
                stage=stage,
                job_id=job,
                indices=indices,
                estimated_task_hours=estimate,
            )
            wait_success(job, event)
            backup()

        submit("labels", initial)
        run(
            "experiments/analyze.py",
            "--labels",
            *[label_files[i] for i in initial],
            "--output",
            gate1_path.parent,
        )
        event("gate1", evidence=decision(gate1_path, 1))
        remaining = [i for i in range(len(label_files)) if i not in initial]
        # Smaller batches release reservations and account measured runtimes.
        chunk_size = max(4, a.max_parallel)
        for offset in range(0, len(remaining), chunk_size):
            submit("labels", remaining[offset : offset + chunk_size])
        submit("forecasters", estimate=0.25, small=True)
        submit("analysis", estimate=0.25, small=True)
        submit("forecast_reports", estimate=0.25, small=True)
        event("gate2", evidence=decision(plan["gate2"], 2))
        for stage in ("online_core", "final_core"):
            # Submit one task first, then use measured runtime for its matched
            # model/task group. Settings and policy choices never change.
            tasks = plan["stages"][stage]
            groups = {}
            for index, task in enumerate(tasks):
                command = task["commands"][0]
                config = json.loads(
                    Path(command[command.index("--config") + 1]).read_text()
                )
                manifest_flag = (
                    "--manifest" if stage == "online_core" else "--final-manifest"
                )
                manifest = command[command.index(manifest_flag) + 1]
                policy = (
                    command[command.index("--policy") + 1]
                    if stage == "online_core"
                    else "final"
                )
                groups.setdefault(
                    (config["model"]["name"], manifest, policy), []
                ).append(index)
            if a.max_parallel > 1:
                # Calibration tasks are predetermined; all finish before the
                # remaining matched runs are reserved from their runtimes.
                first = [indices[0] for indices in groups.values()]
                for offset in range(0, len(first), chunk_size):
                    submit(
                        stage,
                        first[offset : offset + chunk_size],
                        estimate=label_estimate,
                    )
                estimates = {}
                remaining_indices = []
                for indices in groups.values():
                    status = Path(plan["status_root"]) / stage / f"{indices[0]}.json"
                    actual = json.loads(status.read_text())["seconds"] / 3600
                    for index in indices[1:]:
                        estimates[index] = max(0.25, actual * 1.5)
                        remaining_indices.append(index)
                for offset in range(0, len(remaining_indices), chunk_size):
                    batch = remaining_indices[offset : offset + chunk_size]
                    submit(stage, batch, estimate=max(estimates[i] for i in batch))
                continue
            for indices in groups.values():
                submit(stage, indices[:1], estimate=label_estimate)
                status = Path(plan["status_root"]) / stage / f"{indices[0]}.json"
                actual = json.loads(status.read_text())["seconds"] / 3600
                estimate = max(0.25, actual * 1.5)
                for offset in range(1, len(indices), 4):
                    submit(stage, indices[offset : offset + 4], estimate=estimate)
        submit("reports", estimate=0.25, small=True)
        event(
            "core_complete",
            reports=str(run_root / "reports_core"),
            next="Conditional transfer within the remaining measured budget",
        )
        completed_sources = ["Qwen/Qwen3-4B", "unsloth/Llama-3.2-3B-Instruct"]
        from huggingface_hub import HfApi, snapshot_download

        def spent():
            total = 0
            for path in Path(core_plan["status_root"]).glob("*/*.json"):
                if path.parent.name != "submissions":
                    record = json.loads(path.read_text())
                    if record["status"] == "running":
                        raise RuntimeError("A campaign task is still running")
                    hours = record.get("seconds", 0) / 3600
                    total += max(0.25, hours) if a.cluster == "vista" else hours
            return total

        def headroom(incoming, protected=()):
            while True:
                try:
                    check_storage(
                        Path.cwd(),
                        a.storage_quota_gb,
                        headroom_gb=(incoming + 2 * 10**9) / 10**9,
                    )
                    return
                except RuntimeError:
                    eligible = [
                        source
                        for source in completed_sources
                        if source not in protected
                    ]
                    if not eligible:
                        raise
                    source = eligible[0]
                    completed_sources.remove(source)
                    run("experiments/manage_cache.py", "--evict-model", source)
                    event("cache_rotated", completed_model=source)

        for candidate in campaign.get("transfer", []):
            decision(core_plan["gate2"], 2)
            cfg_path = candidate["config"]
            cfg = json.loads(Path(cfg_path).read_text())
            model = cfg["model"]["name"]
            if model not in {
                "google/gemma-3-4b-it",
                "Qwen/Qwen3-8B",
                "meta-llama/Llama-3.1-8B-Instruct",
            }:
                raise ValueError("Unknown transfer backbone")
            slug = model.split("/")[-1].lower()
            estimate = largest_state * 1.5
            check_deadline(campaign, 1, 1, estimate)
            if spent() + estimate > budget:
                raise RuntimeError(
                    "Remaining GPU budget cannot reserve the transfer pilot"
                )
            info = HfApi().model_info(model, files_metadata=True)
            incoming = sum(
                s.lfs.size
                for s in info.siblings
                if s.rfilename.endswith(".safetensors") and s.lfs
            )
            headroom(incoming)
            proof = root / "model_sources" / f"{slug}.json"
            argv = [
                "experiments/prefetch_mirror.py",
                "--official",
                model,
                "--mirror",
                candidate.get("mirror", model),
                "--output",
                str(proof),
            ]
            for donor in candidate.get("metadata_sources", []):
                argv.extend(("--metadata-source", donor))
            run(*argv)
            mapping_path = Path(os.environ["THIRD_EYE_MODEL_SOURCES"])
            mappings = json.loads(mapping_path.read_text())
            mappings[model] = str(proof.resolve())
            write_json(mapping_path, mappings)
            export = (
                f"ALL,THIRD_EYE_CONFIG={Path(cfg_path).resolve()},"
                f"THIRD_EYE_MANIFEST={Path(campaign['data_root']).resolve()}/math/selection.json,"
                f"THIRD_EYE_GATE2={Path(core_plan['gate2']).resolve()}"
            )
            pilot_environment = dict(os.environ)
            pilot_environment.update(
                THIRD_EYE_CONFIG=str(Path(cfg_path).resolve()),
                THIRD_EYE_MANIFEST=str(
                    Path(campaign["data_root"]).resolve() / "math/selection.json"
                ),
                THIRD_EYE_GATE2=str(Path(core_plan["gate2"]).resolve()),
            )
            pilot_argv = ["sbatch", "--parsable", "--time=" + wall_limit(estimate)]
            if a.account:
                pilot_argv.append("--account=" + a.account)
            if a.cluster == "vista":
                pilot_argv.extend(
                    ("--partition=gh", "experiments/jobs/vista_pilot.slurm")
                )
            else:
                pilot_argv.extend(
                    (
                        "--partition=gpu_student",
                        f"--gres={a.gres}",
                        "--export=" + export,
                        "experiments/jobs/pilot.slurm",
                    )
                )
            pilot = scheduler_job_id(
                subprocess.check_output(
                    pilot_argv,
                    text=True,
                    env=pilot_environment,
                )
            )
            event("transfer_pilot_submitted", model=model, job_id=pilot)
            hours = wait_success(pilot, event)
            budget -= hours
            directory = Path("runs/pilot") / pilot
            frozen = root / "frozen" / f"{slug}.json"
            run(
                "experiments/freeze_protocol.py",
                "--config",
                cfg_path,
                "--pilot-run",
                directory,
                "--depth",
                "5",
                "--output",
                frozen,
            )
            ledger = [
                json.loads(line)
                for line in (directory / "ledger.jsonl").read_text().splitlines()
            ]
            label_estimate = (
                max(
                    row["seconds"] / 3600
                    for row in ledger
                    if row.get("status") == "complete"
                )
                * 5
                * 1.25
            )
            if label_estimate > 12:
                raise RuntimeError(
                    "Transfer trajectories exceed the task wall-time ceiling"
                )
            plan_path = build_plan(
                configs,
                campaign["data_root"],
                root / "transfer_plans" / slug,
                run_root,
                transfer_configs=[frozen],
            )
            transfer_plan = json.loads(plan_path.read_text())
            transfer_plan["storage_quota_gb"] = a.storage_quota_gb
            transfer_plan["cluster"] = a.cluster
            if a.cluster == "vista":
                transfer_plan.pop("compute_budget_a100_hours", None)
                transfer_plan["compute_budget_h200_node_hours"] = budget
            stages = {}
            for stage in (
                "transfer_labels",
                "transfer_forecasts",
                "online_transfer",
                "final_transfer",
            ):
                stages[f"{stage}_{slug}"] = transfer_plan["stages"][stage]
            report = transfer_plan["stages"]["reports"][-1]
            for command in report["commands"]:
                command[command.index("--output") + 1] = str(
                    run_root / f"reports_transfer_{slug}"
                )
            stages[f"reports_transfer_{slug}"] = [report]
            transfer_plan["stages"] = stages
            write_json(plan_path, transfer_plan)
            plan = transfer_plan
            submit(f"transfer_labels_{slug}")
            submit(f"transfer_forecasts_{slug}", estimate=0.25, small=True)
            submit(f"online_transfer_{slug}")
            submit(f"final_transfer_{slug}")
            submit(f"reports_transfer_{slug}", estimate=0.25, small=True)
            completed_sources.append(candidate.get("mirror", model))
            event("transfer_complete", model=model)
        if campaign.get("stress", True):
            plan_path, plan = core_plan_path, core_plan
            online_status = Path(plan["status_root"]) / "online_core"
            maximum = max(
                json.loads(path.read_text())["seconds"]
                for path in online_status.glob("*.json")
            )
            estimate = max(0.25, maximum / 3600 * 1.5)
            if spent() + estimate * len(plan["stages"]["stress"]) > budget:
                event(
                    "stress_skipped",
                    reason="Measured reservations exceed remaining GPU budget",
                )
            else:
                mapping = json.loads(
                    Path(os.environ["THIRD_EYE_MODEL_SOURCES"]).read_text()
                )
                protected = {"Qwen/Qwen3-4B", "unsloth/Llama-3.2-3B-Instruct"}
                for config in configs:
                    frozen = json.loads(Path(config).read_text())["model"]
                    proof_path = mapping.get(frozen["name"])
                    source = (
                        json.loads(Path(proof_path).read_text()) if proof_path else None
                    )
                    repo = source["source_repository"] if source else frozen["name"]
                    revision = (
                        source["source_revision"] if source else frozen["revision"]
                    )
                    info = HfApi().model_info(
                        repo, revision=revision, files_metadata=True
                    )
                    size = sum(
                        s.lfs.size
                        for s in info.siblings
                        if s.rfilename.endswith(".safetensors") and s.lfs
                    )
                    headroom(size, protected)
                    snapshot_download(
                        repo,
                        revision=revision,
                        allow_patterns=[
                            "*.json",
                            "*.safetensors",
                            "*.model",
                            "*.jinja",
                            "tokenizer*",
                        ],
                    )
                submit("stress", estimate=estimate)
                submit("reports_stress", estimate=0.25, small=True)
        event("complete", reports=str(run_root), remaining_gpu_hours=budget - spent())
        backup()
    except Exception as exc:
        event("stopped", reason=str(exc))
        try:
            backup()
        except Exception as backup_error:
            event("backup_failed", reason=str(backup_error))
        raise


if __name__ == "__main__":
    main()
