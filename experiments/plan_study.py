"""Materialize a staged, three-seed study from empirically frozen protocols."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.config import Config
from third_eye.io import write_json


def build_plan(
    config_paths,
    data_root,
    output,
    run_root,
    seeds=(42, 43, 44),
    label_trajectories=3,
    transfer_configs=(),
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Study plans are immutable")
    if len(set(seeds)) < 3 or label_trajectories < 1:
        raise ValueError(
            "Core study requires at least three distinct seeds and positive label trajectories"
        )
    paths = [Path(p) for p in config_paths]
    configs = [Config.load(p) for p in paths]
    names = {cfg.model.name for cfg in configs}
    if names != {"Qwen/Qwen3-4B", "meta-llama/Llama-3.2-3B-Instruct"}:
        raise ValueError("Core study requires exactly the two core backbones")
    all_configs = configs + [Config.load(p) for p in transfer_configs]
    allowed_transfer = {
        "google/gemma-3-4b-it",
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
    }
    if any(
        cfg.model.name not in allowed_transfer for cfg in all_configs[len(configs) :]
    ) or len({cfg.model.name for cfg in all_configs}) != len(all_configs):
        raise ValueError(
            "Transfer is restricted to the three distinct held-out backbones"
        )
    for cfg in all_configs:
        if cfg.protocol.status != "frozen" or cfg.model.revision == "main":
            raise ValueError(
                "All study models require measured frozen protocols and pinned revisions"
            )
        if cfg.protocol.utility_weights != configs[0].protocol.utility_weights:
            raise ValueError("All study models must share pre-declared utility weights")
    output.mkdir(parents=True)
    stages = {
        name: []
        for name in (
            "labels",
            "forecasters",
            "forecast_reports",
            "analysis",
            "online_core",
            "online_transfer",
            "transfer_labels",
            "transfer_forecasts",
            "stress",
            "reports_stress",
            "final_core",
            "final_transfer",
            "reports",
        )
    }
    label_paths = []
    forecast_paths = []
    weight_paths = {}
    gate2 = (Path(run_root) / "analysis" / "gate2.json").as_posix()

    def config_file(cfg, name, seed):
        path = output / "configs" / f"{name}_{seed}.json"
        candidate = replace(cfg, protocol=replace(cfg.protocol, seed=seed, depth=5))
        write_json(path, candidate.to_dict())
        return path.as_posix()

    def trajectory(cfg, name, task, seed, mode, policy, stage, requirements=()):
        conf = config_file(cfg, name, seed)
        run = (Path(run_root) / mode / name / task / f"seed{seed}" / policy).as_posix()
        command = [
            "experiments/run.py",
            "--config",
            conf,
            "--manifest",
            (Path(data_root) / task / "selection.json").as_posix(),
            "--mode",
            mode,
            "--policy",
            policy,
            "--output",
            run,
        ]
        if policy in weight_paths:
            command.extend(
                (
                    "--forecaster",
                    (
                        Path(run_root) / "forecasters" / f"{policy}_seed{seed}"
                    ).as_posix(),
                )
            )
        command.extend(("--prune-branches", "--keep-accepted", "1"))
        if "8b" in cfg.model.name.lower():
            command.extend(("--gate2", gate2))
        stages[stage].append(
            {"commands": [command], "requires_gates": list(requirements)}
        )
        return run, conf

    for cfg in configs:
        name = cfg.model.name.split("/")[-1].lower()
        for task in ("math", "code"):
            for seed in seeds:
                for repetition in range(label_trajectories):
                    run, _ = trajectory(
                        cfg,
                        name,
                        task,
                        seed + 1000 * (repetition + 1),
                        "labels",
                        "random",
                        "labels",
                    )
                    label_paths.append(run + "/meta_labels.jsonl")
    for kind, horizon in (
        ("one_step", 1),
        ("matched_h1", 1),
        ("direct", 2),
        ("dynamics", 2),
    ):
        for seed in seeds:
            weight = (Path(run_root) / "forecasters" / f"{kind}_seed{seed}").as_posix()
            command = [
                "experiments/train_forecaster.py",
                "--labels",
                *label_paths,
                "--output",
                weight,
                "--kind",
                kind,
                "--horizon",
                str(horizon),
                "--seed",
                str(seed),
                "--split-seed",
                "42",
                "--device",
                "cuda",
                "--utility-weights",
                *[str(w) for w in configs[0].protocol.utility_weights],
            ]
            stages["forecasters"].append({"commands": [command]})
            forecast_paths.append(weight)
            if seed == seeds[0]:
                weight_paths[kind] = weight
    # Required feature and scalarization ablations reuse the same state split.
    for ablation in (
        "gradient",
        "probe",
        "history",
        "diversity",
        "retention",
        "scalar",
    ):
        weight = (
            Path(run_root) / "forecasters" / f"direct_ablate_{ablation}"
        ).as_posix()
        command = [
            "experiments/train_forecaster.py",
            "--labels",
            *label_paths,
            "--output",
            weight,
            "--kind",
            "direct",
            "--seed",
            str(seeds[0]),
            "--split-seed",
            "42",
            "--device",
            "cuda",
            "--utility-weights",
            *[str(w) for w in configs[0].protocol.utility_weights],
        ]
        command += ["--scalar"] if ablation == "scalar" else ["--ablate", ablation]
        stages["forecasters"].append({"commands": [command]})
        forecast_paths.append(weight)
    stages["forecast_reports"].append(
        {
            "commands": [
                [
                    "experiments/report_forecasts.py",
                    "--models",
                    *forecast_paths,
                    "--baseline",
                    weight_paths["direct"],
                    "--output",
                    (Path(run_root) / "reports_forecasting").as_posix(),
                ]
            ]
        }
    )
    stages["analysis"].append(
        {
            "commands": [
                [
                    "experiments/analyze.py",
                    "--labels",
                    *label_paths,
                    "--forecaster",
                    weight_paths["direct"],
                    "--output",
                    (Path(run_root) / "analysis").as_posix(),
                ]
            ]
        }
    )
    for kind in ("one_step", "matched_h1", "dynamics"):
        stages["analysis"].append(
            {
                "commands": [
                    [
                        "experiments/analyze.py",
                        "--labels",
                        *label_paths,
                        "--forecaster",
                        weight_paths[kind],
                        "--output",
                        (Path(run_root) / f"analysis_{kind}").as_posix(),
                    ]
                ]
            }
        )
    online_runs = []
    final_runs = []
    for cfg in all_configs:
        name = cfg.model.name.split("/")[-1].lower()
        core = cfg in configs
        if not core:
            transfer_paths = []
            for seed in seeds:
                run, _ = trajectory(
                    cfg,
                    name,
                    "math",
                    seed + 900_000,
                    "labels",
                    "random",
                    "transfer_labels",
                    [gate2],
                )
                transfer_paths.append(run + "/meta_labels.jsonl")
            for kind in ("one_step", "direct", "dynamics"):
                stages["transfer_forecasts"].append(
                    {
                        "commands": [
                            [
                                "experiments/analyze.py",
                                "--labels",
                                *transfer_paths,
                                "--forecaster",
                                weight_paths[kind],
                                "--output",
                                (
                                    Path(run_root) / "transfer_forecasts" / name / kind
                                ).as_posix(),
                            ]
                        ],
                        "requires_gates": [gate2],
                    }
                )
        tasks = ("math", "code") if core else ("math",)
        policies = (
            (
                "no_update",
                "random",
                "heuristic",
                "greedy_h1",
                "one_step",
                "matched_h1",
                "direct",
                "dynamics",
            )
            if core
            else ("no_update", "greedy_h1", "direct")
        )
        for task in tasks:
            for seed in seeds:
                for policy in policies:
                    gates = [] if core else [gate2]
                    run, conf = trajectory(
                        cfg,
                        name,
                        task,
                        seed,
                        "online",
                        policy,
                        "online_core" if core else "online_transfer",
                        gates,
                    )
                    online_runs.append(run)
                    final_out = run + "/final_evaluation"
                    command = [
                        "experiments/evaluate_final.py",
                        "--config",
                        conf,
                        "--checkpoint",
                        run + "/accepted/generation_5",
                        "--selection-manifest",
                        (Path(data_root) / task / "selection.json").as_posix(),
                        "--final-manifest",
                        (Path(data_root) / task / "final.json").as_posix(),
                        "--output",
                        final_out,
                    ]
                    if "8b" in cfg.model.name.lower():
                        command.extend(("--gate2", gate2))
                    stages["final_core" if core else "final_transfer"].append(
                        {"commands": [command], "requires_gates": gates}
                    )
                    final_runs.append(final_out)
    core_runs = [
        r
        for r in online_runs
        if any(cfg.model.name.split("/")[-1].lower() in r for cfg in configs)
    ]
    core_finals = [r + "/final_evaluation" for r in core_runs]
    for cfg in configs:
        name = cfg.model.name.split("/")[-1].lower()
        for seed in seeds:
            for policy in ("greedy_h1", "one_step", "direct"):
                run, _ = trajectory(
                    cfg, name, "math", seed + 100_000, "online", policy, "stress"
                )
                command = stages["stress"][-1]["commands"][0]
                command.extend(("--pool-order-seed", str(seed)))
                if policy in weight_paths:
                    # Reuse the preselected seed-matched forecaster; stress
                    # trajectories change task order and update sampling only.
                    command[command.index("--forecaster") + 1] = (
                        Path(run_root) / "forecasters" / f"{policy}_seed{seed}"
                    ).as_posix()
    stages["reports"].append(
        {
            "commands": [
                [
                    "experiments/report_results.py",
                    "--runs",
                    *core_runs,
                    "--final-evaluations",
                    *core_finals,
                    "--output",
                    (Path(run_root) / "reports_core").as_posix(),
                ]
            ]
        }
    )
    stress_runs = [
        t["commands"][0][t["commands"][0].index("--output") + 1]
        for t in stages["stress"]
    ]
    stages["reports_stress"].append(
        {
            "commands": [
                [
                    "experiments/report_results.py",
                    "--runs",
                    *stress_runs,
                    "--output",
                    (Path(run_root) / "reports_stress").as_posix(),
                ]
            ]
        }
    )
    stages["reports"][0]["commands"].append(
        [
            "experiments/report_results.py",
            "--runs",
            *core_runs,
            "--final-evaluations",
            *core_finals,
            "--baseline",
            "one_step",
            "--output",
            (Path(run_root) / "reports_core_vs_h1").as_posix(),
        ]
    )
    if transfer_configs:
        stages["reports"].append(
            {
                "commands": [
                    [
                        "experiments/report_results.py",
                        "--runs",
                        *online_runs,
                        "--final-evaluations",
                        *final_runs,
                        "--output",
                        (Path(run_root) / "reports_all").as_posix(),
                    ]
                ],
                "requires_gates": [gate2],
            }
        )
    plan = {
        "schema_version": 1,
        "seeds": list(seeds),
        "stages": stages,
        "status_root": (Path(run_root) / "task_status").as_posix(),
        "gate2": gate2,
        "config_sources": [str(p) for p in config_paths],
        "data_root": str(data_root),
        "execution_order": [
            "labels",
            "forecasters",
            "forecast_reports",
            "analysis",
            "online_core",
            "final_core",
            "reports",
        ],
        "conditional_stages": [
            "transfer_labels",
            "transfer_forecasts",
            "online_transfer",
            "final_transfer",
        ],
        "optional_stages": ["stress", "reports_stress"],
        "pd": "Skipped until the proposal's measured activation gate passes",
        "compute_budget_a100_hours": 160,
        "storage_quota_gb": 50,
    }
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs=2, required=True)
    p.add_argument("--transfer-configs", nargs="*", default=[])
    p.add_argument("--data-root", default="data/processed/v1")
    p.add_argument("--output", required=True)
    p.add_argument("--run-root", default="runs/study_v1")
    p.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    p.add_argument("--label-trajectories", type=int, default=3)
    a = p.parse_args()
    print(
        build_plan(
            a.configs,
            a.data_root,
            a.output,
            a.run_root,
            a.seeds,
            a.label_trajectories,
            a.transfer_configs,
        )
    )


if __name__ == "__main__":
    main()
