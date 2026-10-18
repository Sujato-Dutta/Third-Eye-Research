"""Compute-only Stage A: corrected harm, admission headroom and precision."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crossed_corrections import fixed_items, rows_for
from third_eye.cluster import require_compute_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.experiments.admission import (
    DECLARATION,
    aggregate,
    arrays,
    conditional_item_bootstrap,
    effective_delta_norm,
    gate_review,
    state_summary,
    synthetic_controls,
)
from third_eye.experiments.crossed import DECLARATION as CROSSED_DECLARATION
from third_eye.io import digest, file_digest, write_json
from third_eye.provenance import source_inventory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_factors(path, adapter_hash, cfg, hashes):
    """Only allocated CPUs load adapter weights. Never load the base model."""
    from safetensors.torch import load_file

    path = Path(path)
    state, adapter = read(path / "state.json"), read(path / "adapter_config.json")
    if state["adapter_hash"] != adapter_hash or state["config_hash"] != cfg.fingerprint:
        raise ValueError("Actual adapter checkpoint identity differs")
    if (
        adapter["r"] != cfg.training.rank
        or adapter["lora_alpha"] != cfg.training.alpha
        or adapter.get("use_dora", False)
        or adapter.get("use_rslora", False)
        or adapter.get("rank_pattern")
        or adapter.get("alpha_pattern")
    ):
        raise ValueError("Norm requires declared standard scaled LoRA")
    tensors = {}
    for name, expected in state["weights"].items():
        weight = path / name
        weight.resolve().relative_to(path.resolve())
        if file_digest(weight) != expected:
            raise ValueError("Checkpoint weights changed")
        hashes[str(weight)] = expected
        if weight.suffix != ".safetensors":
            raise ValueError("Only safe adapter serialization is permitted")
        tensors.update(load_file(str(weight), device="cpu"))
    factors = {}
    for name, tensor in tensors.items():
        if name.endswith(".lora_A.weight"):
            b_name = name.replace(".lora_A.weight", ".lora_B.weight")
            if b_name not in tensors:
                raise ValueError("Adapter B factor missing")
            factors[name] = (tensor.double().numpy(), tensors[b_name].double().numpy())
    if not factors or len(tensors) != 2 * len(factors):
        raise ValueError("Unsupported saved adapter components")
    for name in ("state.json", "adapter_config.json"):
        hashes[str(path / name)] = file_digest(path / name)
    return factors


def load_states(plan):
    """Bind the new analysis to the completed immutable corrected experiment."""
    hashes = {}

    def checked_json(path, expected=None):
        value = file_digest(path)
        if expected is not None and value != expected:
            raise ValueError("Declared historical input changed")
        hashes[str(path)] = value
        return read(path)

    historical = checked_json(plan["crossed_plan"], plan["crossed_plan_sha256"])
    if (
        historical["source_tree_sha256"] != plan["crossed_source_sha256"]
        or historical["declaration"] != CROSSED_DECLARATION
    ):
        raise ValueError("Corrected experiment identity/declaration differs")
    root = Path(historical["output"])
    report = checked_json(root / "report.json", plan["crossed_report_sha256"])
    if (
        report["status"] != "science_review_required"
        or report["missing_tasks"]
        or report["independent_parents"] != 8
    ):
        raise ValueError("Complete corrected eight-parent review required")
    pilot = checked_json(historical["pilot_plan"], historical["pilot_plan_sha256"])
    checked_json(historical["inspection"], historical["inspection_sha256"])
    states = []
    for p, task in enumerate(pilot["tasks"]):
        cfg = Config.load(task["config"])
        _, splits = load_manifest(task["manifest"])
        rows = rows_for(task, pilot)
        labels = Path(task["output"]) / "meta_labels.jsonl"
        hashes[str(labels)] = file_digest(labels)
        hashes[task["manifest"]] = file_digest(task["manifest"])
        hashes[task["config"]] = file_digest(task["config"])
        parent, immediate = None, []
        cube = [[[None] * 3 for _ in range(3)] for _ in range(3)]
        for i in range(3):
            receipt_path = root / f"task_{3 * p + i}/completed.json"
            receipt = checked_json(
                receipt_path, plan["receipts_sha256"][str(3 * p + i)]
            )
            if (
                not receipt.get("passed")
                or receipt["plan_hash"] != digest(historical)
                or receipt["index"] != 3 * p + i
            ):
                raise ValueError("Source-bound corrected completion receipt differs")
            artifacts = {}
            for name, adapter_hash in (
                ("parent", rows[i]["parent_adapter_hash"]),
                ("immediate", rows[i]["candidate_adapter_hash"]),
            ):
                path = receipt[name + "_items"]
                checked_json(path, receipt[name + "_items_sha256"])
                artifacts[name] = fixed_items(path, adapter_hash, splits, cfg)
            if parent is not None and parent != artifacts["parent"]:
                raise ValueError("Repeated fixed-context parent evaluation differs")
            parent = artifacts["parent"]
            if artifacts["immediate"]["input_rendering"] != parent["input_rendering"]:
                raise ValueError("Parent/M1 rendering differs")
            immediate.append(artifacts["immediate"])
            if len(receipt["cells"]) != 9:
                raise ValueError("Nine fresh cells required per recipient")
            for cell in receipt["cells"]:
                j, s = cell["donor_index"], cell["draw_index"]
                if (
                    cell["recipient_index"] != i
                    or j not in range(3)
                    or s not in range(3)
                    or cube[i][j][s] is not None
                    or cell["recipient_adapter_hash"]
                    != rows[i]["candidate_adapter_hash"]
                    or cell["donor"] != rows[j]["candidate_id"]
                    or cell["artifact_kind"]
                    != (
                        "matched_context_native_continuation"
                        if i == j
                        else "counterfactual_crossed_correction"
                    )
                ):
                    raise ValueError("Cell recipient/donor identity differs")
                checked_json(cell["items_path"], cell["items_sha256"])
                cube[i][j][s] = fixed_items(
                    cell["items_path"], cell["continuation_adapter_hash"], splits, cfg
                )
                if cube[i][j][s]["input_rendering"] != parent["input_rendering"]:
                    raise ValueError("Matched-context rendering differs")
        path = Path(rows[0]["checkpoints"]["t1"]).parent.parent / "parent"
        before = load_factors(path, rows[0]["parent_adapter_hash"], cfg, hashes)
        norms = []
        for row in rows:
            after = load_factors(
                row["checkpoints"]["t1"], row["candidate_adapter_hash"], cfg, hashes
            )
            norms.append(
                effective_delta_norm(
                    before, after, cfg.training.alpha / cfg.training.rank
                )
            )
        panel = arrays(parent, immediate, cube)
        states.append(
            {
                "trajectory_id": rows[0]["trajectory_id"],
                "stream": task["stream"],
                "parent": parent,
                "panel": panel,
                "effective_update_norms": norms,
                "summary": state_summary(parent, panel),
            }
        )
        print(f"Validated corrected parent {p + 1}/8: {task['stream']}", flush=True)
    return states, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    require_compute_allocation()
    plan = read(args.plan)
    if (
        plan["declaration"] != DECLARATION
        or plan["source_tree_sha256"] != source_inventory()["source_tree_sha256"]
    ):
        raise ValueError("Stage A declaration or current immutable source changed")
    validation = read(plan["validation"])
    if (
        validation.get("passed") is not True
        or validation["source_tree_sha256"] != plan["source_tree_sha256"]
    ):
        raise ValueError("Full source-bound CPU validation must pass")
    if file_digest("research.md") != plan["research_sha256"]:
        raise ValueError("Authorized research plan changed")
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    write_json(output / "declaration.json", plan)
    try:
        states, hashes = load_states(plan)
        gate_fits = gate_review(states)
        result = aggregate(states, gate_fits)
        result["conditional_item_uncertainty"] = conditional_item_bootstrap(
            states, gate_fits
        )
        result["synthetic_controls"] = synthetic_controls()
        result["feature_audit"] = {
            "used": DECLARATION["features"],
            "excluded": "all historical text/gradient probes: no verified fixed-rendering context",
            "probe_adapter_delta_norm_not_used": "factor norm after short probe is not actual effective committed update",
            "source_exclusion_flags_preserved": True,
            "off_diagonal_cells_used_in_fitting": False,
        }
        result["per_parent"] = [
            {
                k: s[k]
                for k in (
                    "trajectory_id",
                    "stream",
                    "effective_update_norms",
                    "summary",
                )
            }
            for s in states
        ]
        audit_path = Path("experiments/admission_precision_audit.json")
        result["related_work_precision_audit"] = read(audit_path)
        hashes[str(audit_path)] = file_digest(audit_path)
        write_json(output / "gate_fits.json", gate_fits)
        write_json(output / "report.json", result)
        with (output / "parent_metrics.csv").open("w", newline="") as handle:
            fields = ["trajectory_id", "stream", *states[0]["summary"]["metrics"]]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for state in states:
                writer.writerow(
                    {
                        "trajectory_id": state["trajectory_id"],
                        "stream": state["stream"],
                        **state["summary"]["metrics"],
                    }
                )
        with (output / "related_work_precision.csv").open("w", newline="") as handle:
            papers = result["related_work_precision_audit"]["papers"]
            writer = csv.DictWriter(handle, fieldnames=list(papers[0]))
            writer.writeheader()
            writer.writerows(papers)
        write_json(
            output / "completed.json",
            {
                "passed": True,
                "plan_hash": digest(plan),
                "report_sha256": file_digest(output / "report.json"),
                "input_hashes": hashes,
                "source_tree_sha256": plan["source_tree_sha256"],
                "automatic_expansion": False,
            },
        )
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "metrics": result["metrics"],
                    "proxy_minus_better_constant": result[
                        "proxy_minus_better_constant"
                    ],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "plan_hash": digest(plan),
            },
        )
        raise


if __name__ == "__main__":
    main()
