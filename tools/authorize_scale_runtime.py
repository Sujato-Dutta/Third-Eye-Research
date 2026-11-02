"""Allocated timing supplement for the frozen eight-parent Llama extension.

Preserve the failed sixteen-prompt extrapolation. Measure full-budget math
harvesting on discarded parent/updated fixtures before authorizing collection.
No calibration corrections or evaluation outcomes enter the research cohort.
"""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path


def covered_costs(original, measurements, sizes, bounds):
    if set(bounds) != {"code", "math"} or set(sizes) != set(bounds):
        raise ValueError("Both fixed task reservations required")
    if any(type(s) is not int or not 0 < s <= 172800 for s in bounds.values()):
        raise ValueError("Reservations must fit Vista's48-hour QOS")
    by_task = {row["task"]: deepcopy(row) for row in original}
    if len(original) != 2 or set(by_task) != set(bounds):
        raise ValueError("Complete original both-task compatibility timings required")
    if len(measurements) != 2 or {m["state"] for m in measurements} != {"parent", "trained"}:
        raise ValueError("Parent and trained full-pool math timings required")
    for item in measurements:
        if (item["training_prompts"] != sizes["math"] or item["harvest_complete"] is not True
                or item["attempt_limit"] != 8 or item["seed_stride"] != 16
                or not 0 < item["seconds"] < float("inf")):
            raise ValueError("Complete unchanged math harvest budget required")
    for task, row in by_task.items():
        if row["margin"] != 1.5 or row["includes_three_continuations_per_candidate"] is not True or row["scored_accuracy_recorded"] is not False:
            raise ValueError("Unchanged complete-protocol timing required")
        timings = row["per_item_final_timing"]
        scoring = 13 * sum(max(timings[r]) * n for r, n in
                           [("target_test", 192), ("ood_test", 192), ("retention_test", 512)])
        estimated = 1.5 * (scoring + 10 * sizes[task] * row["discarded16_prompt_harvest_seconds"] / 16
                           + 24 * row["train_seconds"] + 3600)
        if abs(estimated - row["required_seconds"]) > 1e-5:
            raise ValueError("Original complete-runtime formula differs")
        row["original_required_seconds"] = row["required_seconds"]
        row["original_reserved_seconds"] = row["reserved_seconds"]
        if task == "math":
            row["full_pool_harvest_seconds"] = max(m["seconds"] for m in measurements)
            row["required_seconds"] = 1.5 * (scoring + 10 * row["full_pool_harvest_seconds"]
                                              + 24 * row["train_seconds"] + 3600)
            row["pricing_basis"] = "maximum_of_two_complete_1024_prompt_harvests"
        else:
            row["pricing_basis"] = "original_complete_protocol_extrapolation"
        row["reserved_seconds"] = bounds[task]
        if not 0 < row["required_seconds"] <= bounds[task]:
            raise ValueError("Measured complete protocol exceeds declared task walltime")
    return [by_task[t] for t in ("code", "math")]


def self_test():
    import unittest

    class RuntimeTests(unittest.TestCase):
        def fixture(self):
            rows = []
            for task, size in [("code", 306), ("math", 1024)]:
                row = {"task": task, "margin": 1.5, "includes_three_continuations_per_candidate": True,
                       "scored_accuracy_recorded": False, "per_item_final_timing": {r: [1, 2] for r in ("target_test", "ood_test", "retention_test")},
                       "discarded16_prompt_harvest_seconds": 200 if task == "code" else 250, "train_seconds": 150, "reserved_seconds": 115200}
                row["required_seconds"] = 1.5 * (13 * 2 * (192 + 192 + 512) + 10 * size * row["discarded16_prompt_harvest_seconds"] / 16 + 24 * 150 + 3600)
                rows.append(row)
            measurements = [{"state": state, "training_prompts": 1024, "harvest_complete": True,
                             "attempt_limit": 8, "seed_stride": 16, "seconds": seconds}
                            for state, seconds in [("parent", 2000), ("trained", 3000)]]
            return rows, measurements, {"code": 306, "math": 1024}, {"code": 115200, "math": 172800}

        def test_full_pool_timing_replaces_only_math_extrapolation(self):
            rows, measurements, sizes, bounds = self.fixture()
            result = covered_costs(rows, measurements, sizes, bounds)
            self.assertEqual(result[0]["required_seconds"], rows[0]["required_seconds"])
            self.assertLess(result[1]["required_seconds"], rows[1]["required_seconds"])
            self.assertEqual(result[1]["full_pool_harvest_seconds"], 3000)
            self.assertEqual(rows[1]["reserved_seconds"], 115200)

        def test_incomplete_harvest_cannot_authorize_collection(self):
            rows, measurements, sizes, bounds = self.fixture()
            measurements[1]["training_prompts"] = 16
            with self.assertRaises(ValueError):
                covered_costs(rows, measurements, sizes, bounds)

        def test_runtime_and_site_limits_remain_enforced(self):
            rows, measurements, sizes, bounds = self.fixture()
            measurements[1]["seconds"] = 100000
            with self.assertRaises(ValueError):
                covered_costs(rows, measurements, sizes, bounds)
            bounds["math"] = 172801
            with self.assertRaises(ValueError):
                covered_costs(rows, measurements, sizes, bounds)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def measure(path):
    import torch
    from scale_confirmation import checked, plan_for, read
    from third_eye.cluster import require_gpu_allocation
    from third_eye.config import Config
    from third_eye.data.schema import Correction
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.runner import verify_completions
    from third_eye.evaluation.verifiers import VerifierRegistry
    from third_eye.experiments.scale import DECLARATION
    from third_eye.io import digest, file_digest, write_json
    from third_eye.training.hf_backend import HFBackend
    from third_eye.updates.corrections import collect_corrections

    require_gpu_allocation()
    auth = read(path)
    if (file_digest(__file__) != auth["operational_script_sha256"]
            or auth["scope"] != "existing_eight_llama_parents_only"
            or auth["prior_revision_cap_removed_for_llama"] is not True
            or auth["scientific_settings_changed"] is not False):
        raise ValueError("Matching explicit Llama placement/runtime authorization required")
    for entry in auth["frozen_inputs"].values():
        if file_digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen input changed")
    request = checked(auth["request"])
    model = "llama3.1-8b"
    plan = plan_for(request, model)
    if plan["plan_sha256"] != auth["plan_sha256"] or len(plan["tasks"]) != 8:
        raise ValueError("Original balanced eight-parent plan required")
    ledger = read(auth["ledger"])
    total = sum(j.get("actual_sus", j["reserved_sus"]) for j in ledger["jobs"].values())
    if total > ledger["hard_cap_sus"] + 1e-8:
        raise ValueError("Scoped authorized reservations exceeded")
    for task, seconds in auth["seconds_by_task"].items():
        job = ledger["jobs"][auth["reservation_names"][task]]
        if (job["partition"] != "gh" or job["seconds"] != seconds or job["tasks"] != 4
                or job["job_id"] != auth["collection_jobs"][task]):
            raise ValueError("Actual submitted parent reservations differ")
    directory = Path(auth["output"])
    directory.mkdir(exist_ok=False)
    original = read(auth["frozen_inputs"]["original_cost"]["path"])
    sizes = {}
    for task in ("code", "math"):
        item = next(t for t in plan["tasks"] if t["stream"].endswith("/" + task))
        _, splits = load_manifest(item["manifest"])
        sizes[task] = len(splits["train"])
    if sizes != {"code": 306, "math": 1024}:
        raise ValueError("Original complete training pools required")
    task = next(t for t in plan["tasks"] if t["stream"].endswith("/math"))
    cfg = Config.load(task["config"])
    _, splits = load_manifest(task["manifest"])
    backend = HFBackend(cfg)
    verifier = VerifierRegistry.from_spec("third_eye.evaluation.benchmarks:make_verifier")
    parent, original_hash = backend.snapshot(), backend.state_hash()
    fixtures = [Correction(e, e.answer, 1) for e in splits["train"][:2]]
    if not all(verify_completions(verifier, [c.example for c in fixtures], [c.completion for c in fixtures])):
        raise ValueError("Discarded native gold fixtures must verify")
    measurements = []
    for state in ("parent", "trained"):
        if state == "trained":
            training = backend.train(fixtures, cfg.protocol.seed + 910_000_000)
            if training["optimizer_steps"] != 50 or backend.state_hash() == original_hash:
                raise ValueError("Discarded fixture update failed")
        pool, stats = collect_corrections(backend, splits["train"], verifier, cfg.protocol,
                                          cfg.protocol.seed + 900_000_000)
        measurements.append({"state": state, "seconds": stats["total_seconds"],
                             "training_prompts": stats["training_prompts"], "harvest_complete": stats["harvest_complete"],
                             "attempt_limit": stats["correction_attempt_limit"], "seed_stride": stats["correction_seed_stride"],
                             "corrections_discarded": True, "accuracy_recorded": False})
        del pool
        write_json(directory / "timing_progress.json", measurements)
    backend.restore(parent)
    if backend.state_hash() != original_hash:
        raise ValueError("Timing fixture update persisted")
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    write_json(directory / "full_pool_timings.json", measurements)
    revised = covered_costs(original, measurements, sizes, auth["seconds_by_task"])
    output = Path(request["output"]) / model
    ready = output / "preflight/completed.json"
    permission = output / "authorized_scale.json"
    if ready.exists() or permission.exists():
        raise FileExistsError("Preserve previous readiness/authorization")
    record = {"passed": True, "source_tree_sha256": request["source_tree_sha256"],
              "plan_sha256": plan["plan_sha256"], "model": model, "discarded_fixture": True,
              "runtime_coverage": revised, "full_pool_timings": str(directory / "full_pool_timings.json"),
              "original_runtime_failure_preserved": True, "scientific_settings_changed": False,
              "operational_authorization_sha256": file_digest(path),
              "operational_script_sha256": file_digest(__file__)}
    write_json(ready, record)
    write_json(permission, {"purpose": "frozen_empirical_scale_confirmation", "model": model,
                           "declaration_sha256": digest(DECLARATION), "source_tree_sha256": request["source_tree_sha256"],
                           "preflight": str(ready), "preflight_sha256": file_digest(ready),
                           "tasks": [{"seed": t["seed"], "config_hash": Config.load(t["config"]).fingerprint,
                                      "manifest_sha256": t["manifest_sha256"]} for t in plan["tasks"]],
                           "forecasting_gate_passed_claimed": False})
    write_json(directory / "completed.json", record)
    print(json.dumps({"complete_runtime_verified": True, "costs": revised}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    if args.self_test:
        from third_eye.cluster import require_compute_allocation
        require_compute_allocation()
        self_test()
    elif args.authorization:
        measure(args.authorization)
    else:
        parser.error("Choose --self-test or --authorization")


if __name__ == "__main__":
    main()
