"""Start one Vista campaign after compute-node preflight succeeds."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from continue_study import wait_success
from third_eye.io import append_jsonl, write_json
from third_eye.cluster import scheduler_job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-job", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--max-parallel", type=int, required=True)
    parser.add_argument("--budget-hours", type=float, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        parser.error("Campaign launch requires a scheduled compute node")
    if not 1 <= args.max_parallel <= 20 or args.budget_hours <= 0:
        parser.error("Parallelism must fit the verified Vista limit and budget")
    if (
        Path(args.campaign).exists()
        or Path(args.campaign + ".launch.json").exists()
        or Path(args.output).exists()
    ):
        parser.error("Campaign manifests and outputs must be new")
    launch_events = Path(args.output + "_launch.jsonl")

    def event(status, **details):
        record = {"status": status, **details}
        append_jsonl(launch_events, record)
        print(json.dumps(record), flush=True)

    spent = max(0.25, wait_success(args.preflight_job, event))
    proof = Path("runs/vista_preflight") / args.preflight_job / "passed.json"
    preflight = json.loads(proof.read_text())
    if preflight.get("status") != "passed":
        raise RuntimeError("Vista preparation has not passed")
    cpu_setup = preflight.get("cpu_setup_job")
    cpu_hours = max(0.25, wait_success(cpu_setup, event)) if cpu_setup else 0
    profile = Path(f"runs/vista_gpu_profile_{args.preflight_job}.json")
    if json.loads(profile.read_text()).get("status") != "passed":
        raise RuntimeError("Vista GPU profile has not passed")
    # Four maximum pilot allocations are reserved before any is submitted.
    if spent + 4 * 12 > args.budget_hours:
        raise RuntimeError("Budget cannot reserve the four measured pilots")
    environment = dict(os.environ)
    environment.update(
        THIRD_EYE_MODEL_SOURCES=str(
            Path("experiments/model_sources_vista.json").resolve()
        ),
        THIRD_EYE_SANDBOX_CONFIG=str(Path("experiments/sandbox_vista.json").resolve()),
        THIRD_EYE_VERIFIER="third_eye.evaluation.benchmarks:make_verifier",
        THIRD_EYE_VERIFIER_WORKERS="16",
        THIRD_EYE_SYMBOLIC_WORKERS="1",
    )
    pilots = []
    for model, slug in (
        ("Qwen/Qwen3-4B", "qwen3_4b"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama3_2_3b"),
    ):
        for task in ("math", "code"):
            config = f"experiments/configs/{slug}_vista_pilot.json"
            pilot_environment = dict(environment)
            pilot_environment.update(
                THIRD_EYE_CONFIG=str(Path(config).resolve()),
                THIRD_EYE_MANIFEST=str(
                    Path(f"data/processed/v1/{task}/selection.json").resolve()
                ),
            )
            job = scheduler_job_id(
                subprocess.check_output(
                    [
                        "sbatch",
                        "--parsable",
                        "--partition=gh",
                        "--account=" + args.account,
                        "experiments/jobs/vista_pilot.slurm",
                    ],
                    env=pilot_environment,
                    text=True,
                )
            )
            pilot = {
                "model": model,
                "task": task,
                "config": config,
                "job_id": job,
                "run": f"runs/pilot/{job}",
            }
            pilots.append(pilot)
            event("pilot_submitted", **pilot)
            # Persist each ID immediately so a later submission failure cannot
            # leave an unrecorded allocation or duplicate submission on retry.
            write_json(
                args.campaign + ".launch.json",
                {"pilots": pilots, "preflight_job": args.preflight_job},
            )
    campaign = {
        "cluster": "vista",
        "account": args.account,
        "pilots": pilots,
        "data_root": "data/processed/v1",
        "prior_gpu_hours": spent,
        "compute_budget_hours": args.budget_hours,
        "compute_budget_unit": "Vista H200 node-hours; 15-minute minimum per job",
        "storage_quota_gb": 50,
        "max_parallel": args.max_parallel,
        "preflight_job": args.preflight_job,
        "preflight_proof": str(proof),
        "cpu_setup_job": cpu_setup,
        "cpu_setup_node_hours": cpu_hours,
        "cpu_setup_service_units": cpu_hours * 0.33,
        "previous_campaign": {
            "cluster": "dgx",
            "manifest": "experiments/campaign_20261001_v6.json",
            "purpose": "Archived calibration; excluded from Vista study labels",
            "budget_units": "A100 GPU-hours, recorded separately",
        },
        "target_completion_date": "2026-10-10",
        "completion_deadline_utc": "2026-10-10T18:29:59+00:00",
        "stress": True,
        "transfer": [
            {
                "config": "experiments/configs/gemma3_4b_vista_pilot.json",
                "mirror": "unsloth/gemma-3-4b-it",
                "metadata_sources": ["mlx-community/gemma-3-4b-it-bf16"],
            },
            {"config": "experiments/configs/qwen3_8b_vista_pilot.json"},
            {
                "config": "experiments/configs/llama3_1_8b_vista_pilot.json",
                "mirror": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "metadata_sources": ["NousResearch/Meta-Llama-3.1-8B-Instruct"],
            },
        ],
    }
    write_json(args.campaign, campaign)
    subprocess.run(
        [
            sys.executable,
            "experiments/continue_study.py",
            "--cluster",
            "vista",
            "--account",
            args.account,
            "--max-parallel",
            str(args.max_parallel),
            "--budget-hours",
            str(args.budget_hours),
            "--campaign",
            args.campaign,
            "--output",
            args.output,
        ],
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
