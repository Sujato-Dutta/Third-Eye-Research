"""Check deployed workflow arguments on a CPU allocation without executing tasks.

Plans here are explicitly operational fixtures, never frozen research
protocols, training labels, or passing gate evidence.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import runpy
import socket
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]


class ArgumentsChecked(BaseException):
    pass


def check_command(command, modules):
    script = command[0]
    if script not in modules:
        modules[script] = runpy.run_path(str(ROOT / script))
    original = argparse.ArgumentParser.parse_args

    def parse_only(parser, args=None, namespace=None):
        original(parser, args, namespace)
        raise ArgumentsChecked

    with (
        patch.object(sys, "argv", command),
        patch.object(argparse.ArgumentParser, "parse_args", parse_only),
    ):
        try:
            modules[script]["main"]()
        except ArgumentsChecked:
            return
    raise RuntimeError("CLI did not parse its task arguments: " + script)


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Operational audit requires a compute allocation")
    from third_eye.config import Config
    from third_eye.io import write_json
    from third_eye.provenance import source_inventory
    from third_eye.training.sources import resolve_source
    from plan_study import build_plan

    output = ROOT / "runs/vista_operational_audit" / os.environ["SLURM_JOB_ID"]
    output.mkdir(parents=True, exist_ok=False)
    proof = json.loads((ROOT / "runs/vista_preflight/1040277/passed.json").read_text())
    inventory = source_inventory(ROOT)
    if inventory["source_tree_sha256"] != proof["source_tree_sha256"]:
        raise RuntimeError("Research source changed after GPU preflight")
    core, transfer = [], []
    for name in ("qwen3_4b", "llama3_2_3b", "gemma3_4b", "qwen3_8b", "llama3_1_8b"):
        cfg = Config.load(ROOT / f"experiments/configs/{name}_vista_pilot.json")
        if name in ("qwen3_4b", "llama3_2_3b"):
            resolve_source(cfg.model)
        # Fixture-only revisions permit testing the actual study planner;
        # these configs never enter the research campaign.
        fixture = replace(
            cfg,
            model=replace(cfg.model, revision="a" * 40),
            protocol=replace(cfg.protocol, status="frozen", depth=5),
        )
        path = output / "fixtures" / (name + ".json")
        write_json(path, fixture.to_dict())
        (core if name in ("qwen3_4b", "llama3_2_3b") else transfer).append(path)
    plan_path = build_plan(
        core,
        ROOT / "data/processed/v1",
        output / "fixture_plan",
        output / "fixture_results",
        transfer_configs=transfer,
    )
    plan = json.loads(plan_path.read_text())
    modules, checked = {}, 0
    for tasks in plan["stages"].values():
        for task in tasks:
            for command in task["commands"]:
                check_command(command, modules)
                checked += 1
    write_json(
        output / "passed.json",
        {
            "status": "passed",
            "purpose": "Operational validation only; not research gate evidence",
            "job_id": os.environ["SLURM_JOB_ID"],
            "commands_checked": checked,
            "stages": {k: len(v) for k, v in plan["stages"].items()},
            "cli_entrypoints": sorted(modules),
            "max_gpu_parallelism": 20,
            **inventory,
        },
    )
    print(f"PASS: {checked} actual CLI argument sets across {len(modules)} entrypoints")


if __name__ == "__main__":
    main()
