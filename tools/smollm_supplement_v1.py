# ruff: noqa: E402
"""S1: independent four-state SmolLM3 confirmation, without changing E2."""

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_common_v1 import BASE, prepared
import empirical_checkpoint_v3 as core
from third_eye.cluster import require_gpu_allocation
from third_eye.config import Config
from third_eye.io import file_digest, write_json

OUT = BASE / "smollm_supplement_v1"
MODEL = "HuggingFaceTB/SmolLM3-3B"


def frozen():
    core.frozen()
    release = json.loads((OUT / "release.json").read_text())
    if release["core_release_sha256"] != file_digest(core.OUT / "release.json"):
        raise RuntimeError("S1 core release changed")
    if any(file_digest(ROOT / p) != sha for p, sha in release["files"].items()):
        raise RuntimeError("S1 implementation or scope changed")
    return release["scope"]


def make_config(original, revision, seed):
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or seed not in (6142, 6143):
        raise ValueError("Pinned official revision and fixed seed required")
    return replace(
        original,
        model=replace(
            original.model,
            name=MODEL,
            revision=revision,
            chat_kwargs={"enable_thinking": False},
        ),
        protocol=replace(original.protocol, seed=seed),
    )


def setup():
    frozen()
    from huggingface_hub import HfApi, snapshot_download

    out = OUT / "setup"
    out.mkdir(exist_ok=False)
    try:
        info = HfApi().model_info(MODEL, files_metadata=True, token=False)
        if info.gated or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
            raise RuntimeError(
                "Expected public official SmolLM3 repository and fixed commit"
            )
        revision = info.sha
        snapshot_download(
            MODEL,
            revision=revision,
            token=False,
            allow_patterns=["*.json", "*.model", "*.jinja", "*.txt"],
            max_workers=4,
        )
        from transformers import AutoConfig, AutoTokenizer
        from third_eye.data.splits import load_manifest
        from third_eye.training.tokenization import encode_completion

        config_object = AutoConfig.from_pretrained(
            MODEL, revision=revision, local_files_only=True
        )
        if config_object.model_type != "smollm3":
            raise RuntimeError("Unexpected architecture in official SmolLM3 snapshot")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=revision, use_fast=True, local_files_only=True
        )
        if not tokenizer.is_fast:
            raise RuntimeError("Precise assistant masking needs a fast tokenizer")
        examples = prepared()["replication_tasks"]
        templates = {
            f: next(t for t in examples if t["stream"] == "qwen3-4b/" + f)
            for f in ("code", "math")
        }
        contracts = []
        for family, template in templates.items():
            cfg = make_config(Config.load(template["config"]), revision, 6142)
            _, splits = load_manifest(template["manifest"])
            for example in splits["train"][:2]:
                encoded = encode_completion(
                    tokenizer,
                    example.prompt,
                    example.answer,
                    cfg.training.max_sequence_length,
                    cfg.model.chat_kwargs,
                )
                contracts.append(
                    dict(
                        task_family=family,
                        id=example.id,
                        tokens=len(encoded["input_ids"]),
                        supervised_tokens=sum(v != -100 for v in encoded["labels"]),
                    )
                )
        write_json(
            out / "tokenizer_contracts.json",
            dict(
                model_type=config_object.model_type,
                fixtures=contracts,
                purpose="Compatibility only; gold references do not enter scientific updates",
            ),
        )
        local = Path(
            snapshot_download(
                MODEL,
                revision=revision,
                token=False,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.jinja",
                    "*.txt",
                ],
                max_workers=4,
            )
        )
        checksums, official = {}, {}
        for sibling in info.siblings:
            path = local / sibling.rfilename
            if not path.is_file():
                continue
            actual = file_digest(path)
            lfs = getattr(sibling, "lfs", None)
            expected = (
                lfs.get("sha256")
                if isinstance(lfs, dict)
                else getattr(lfs, "sha256", None)
            )
            if expected and actual != expected:
                raise RuntimeError(
                    f"Official weight checksum differs: {sibling.rfilename}"
                )
            checksums[str(path)] = actual
            if expected:
                official[str(path)] = expected
        if not any(p.endswith(".safetensors") for p in checksums):
            raise RuntimeError("No official trainable weights downloaded")
        write_json(
            OUT / "model_identity.json",
            dict(
                model=MODEL,
                revision=revision,
                snapshot=str(local),
                file_sha256=checksums,
                official_lfs_sha256=official,
                selected_before_scientific_outcomes=True,
            ),
        )
        tasks = []
        for task_family in ("code", "math"):
            template = next(
                t for t in examples if t["stream"] == "qwen3-4b/" + task_family
            )
            original = Config.load(template["config"])
            for seed in (6142, 6143):
                cfg = make_config(original, revision, seed)
                path = OUT / "configs" / f"{task_family}_{seed}.json"
                write_json(path, cfg.to_dict())
                tasks.append(
                    dict(
                        index=len(tasks),
                        task_family=task_family,
                        stream="smollm3-3b/" + task_family,
                        seed=seed,
                        config=str(path),
                        config_sha256=file_digest(path),
                        manifest=template["manifest"],
                        manifest_sha256=template["manifest_sha256"],
                        output=str(OUT / "labels" / task_family / f"seed{seed}"),
                        generations=1,
                    )
                )
        write_json(
            OUT / "tasks.json",
            dict(
                tasks=tasks,
                model_identity_sha256=file_digest(OUT / "model_identity.json"),
                release_sha256=file_digest(OUT / "release.json"),
            ),
        )
        write_json(
            OUT / "cpu_passed.json",
            dict(
                status="passed",
                release_sha256=file_digest(OUT / "release.json"),
                tasks_sha256=file_digest(OUT / "tasks.json"),
            ),
        )
    except Exception as exc:
        write_json(
            out / "failure.json", dict(error_type=type(exc).__name__, error=str(exc))
        )
        raise


def tasks(check_weights=False):
    frozen()
    receipt = json.loads((OUT / "cpu_passed.json").read_text())
    record = json.loads((OUT / "tasks.json").read_text())
    if (
        receipt["status"] != "passed"
        or receipt["release_sha256"] != file_digest(OUT / "release.json")
        or receipt["tasks_sha256"] != file_digest(OUT / "tasks.json")
    ):
        raise RuntimeError("S1 setup must pass without changing its tasks")
    if record["model_identity_sha256"] != file_digest(OUT / "model_identity.json"):
        raise RuntimeError("S1 official model identity changed")
    identity = json.loads((OUT / "model_identity.json").read_text())
    if check_weights and any(
        file_digest(p) != sha for p, sha in identity["file_sha256"].items()
    ):
        raise RuntimeError("Pinned SmolLM3 snapshot changed")
    ts = record["tasks"]
    if len(ts) != 4 or {(t["task_family"], t["seed"]) for t in ts} != {
        (f, s) for f in ("math", "code") for s in (6142, 6143)
    }:
        raise RuntimeError("S1 fixed four-state grid changed")
    for task in ts:
        if (
            file_digest(task["config"]) != task["config_sha256"]
            or file_digest(task["manifest"]) != task["manifest_sha256"]
        ):
            raise RuntimeError("S1 config/manifest changed")
        cfg = Config.load(task["config"])
        if (
            cfg.model.name != MODEL
            or cfg.model.revision != identity["revision"]
            or cfg.protocol.seed != task["seed"]
            or task["generations"] != 1
        ):
            raise RuntimeError("S1 model or bounded invocation changed")
    return ts


def verify():
    scope = frozen()
    require_gpu_allocation()
    core.passed("cpu")
    core.passed("cuda")
    core.arm_deadline(scope)
    from third_eye.training.hf_backend import HFBackend
    from third_eye.data.schema import Correction
    from third_eye.data.splits import load_manifest
    from third_eye.evaluation.benchmarks import make_verifier
    from third_eye.evaluation.runner import verify_completions
    from third_eye.experiments.features import extract_features
    from empirical_checkpoint_noise_v3 import collect_items
    import torch

    target = OUT / "cuda_verification"
    target.mkdir(exist_ok=False)
    results = []
    try:
        for task in tasks(check_weights=True)[::2]:
            cfg = Config.load(task["config"])
            _, splits = load_manifest(task["manifest"])
            verifier = make_verifier()
            exs = splits["train"][:2]
            completions = [ex.answer for ex in exs]
            if not all(verify_completions(verifier, exs, completions)):
                raise RuntimeError("Operational gold train fixture failed verifier")
            batch = tuple(Correction(ex, text, 1) for ex, text in zip(exs, completions))
            backend = HFBackend(cfg)
            parent, before = backend.snapshot(), backend.state_hash()
            features = extract_features(
                backend,
                batch,
                splits["retention_dev"][:2],
                cfg.protocol,
                cfg.protocol.seed + 300_000,
                target / f"{task['task_family']}_probe.jsonl",
            )
            if backend.state_hash() != before:
                raise RuntimeError("SmolLM3 feature probe failed rollback")
            log = backend.train(
                batch,
                cfg.protocol.seed + 300_000,
                log_path=target / f"{task['task_family']}_training.jsonl",
            )
            if log["optimizer_steps"] != 50 or backend.state_hash() == before:
                raise RuntimeError(
                    "SmolLM3 optimization is not a genuine 50-step update"
                )
            trained = backend.state_hash()
            checkpoint = target / task["task_family"] / "updated"
            backend.save_checkpoint(checkpoint, dict(operational_validation_only=True))
            backend.restore(parent)
            backend.load_checkpoint(checkpoint)
            if backend.state_hash() != trained:
                raise RuntimeError("SmolLM3 saved adapter failed exact roundtrip")
            backend.restore(parent)
            if backend.state_hash() != before:
                raise RuntimeError("SmolLM3 parent rollback differs")
            tiny = {r: exs[:2] for r, exs in splits.items() if r.endswith("_dev")}
            measured = collect_items(
                backend, tiny, verifier, cfg.protocol, cfg.protocol.seed
            )
            if measured["adapter_hash"] != before:
                raise RuntimeError("SmolLM3 evaluation changed weights")
            from empirical_replication_v1 import methods
            from third_eye.forecasting.training import Forecaster
            import numpy as np

            views = [
                dict(
                    candidate_id=f"k{i}",
                    precommit=dict(
                        state=measured["metrics"],
                        features=features,
                        history=[],
                        pool_statistics={},
                        generation=0,
                    ),
                )
                for i in range(3)
            ]
            shapes = {}
            for name in ("original_direct_seed42", "original_matched_h1_seed42"):
                spec = next(m for m in methods()["models"] if m["name"] == name)
                prediction = Forecaster.load(spec["folder"]).predict(views)
                if prediction.shape[0] != 3 or not np.isfinite(prediction).all():
                    raise RuntimeError(
                        "Frozen forecast schema failed compatibility-only inputs"
                    )
                shapes[name] = list(prediction.shape)
            results.append(
                dict(
                    task_family=task["task_family"],
                    optimizer_steps=50,
                    probe_steps=10,
                    checkpoint_roundtrip=True,
                    parent_rollback=True,
                    terminal_parent_preservation=True,
                    metrics=measured["metrics"],
                    features=features,
                    frozen_forecast_shapes=shapes,
                    fixture="Gold train completions for compatibility only; discarded and excluded from scientific labels",
                )
            )
            del backend, parent
            gc.collect()
            torch.cuda.empty_cache()
        write_json(
            OUT / "cuda_passed.json",
            dict(
                status="passed",
                cases=results,
                release_sha256=file_digest(OUT / "release.json"),
                tasks_sha256=file_digest(OUT / "tasks.json"),
            ),
        )
    except Exception as exc:
        write_json(
            target / "failure.json", dict(error_type=type(exc).__name__, error=str(exc))
        )
        raise


def run(index):
    scope = frozen()
    require_gpu_allocation()
    core.arm_deadline(scope)
    receipt = json.loads((OUT / "cuda_passed.json").read_text())
    if (
        receipt["status"] != "passed"
        or receipt["release_sha256"] != file_digest(OUT / "release.json")
        or receipt["tasks_sha256"] != file_digest(OUT / "tasks.json")
    ):
        raise RuntimeError("S1 both-task CUDA validation required")
    task = tasks(check_weights=True)[index]
    if Path(task["output"]).exists():
        raise RuntimeError(
            "Retain every attempted run; never overwrite or silently restart"
        )
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
        "1",
        "--keep-accepted",
        "5",
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")


def review():
    frozen()
    from a2_after_labels import audit_labels
    from empirical_cached_v1 import selection
    from empirical_replication_v1 import methods, future_selection, stratified_interval
    from third_eye.forecasting.training import Forecaster
    import numpy as np

    output = OUT / "review"
    output.mkdir(exist_ok=False)
    rows, coverage = [], []
    for task in tasks():
        source = Path(task["output"])
        if not (source / "completed.json").exists():
            coverage.append(dict(index=task["index"], status="incomplete", states=0))
            continue
        completed = json.loads((source / "completed.json").read_text())
        records = audit_labels(source, Config.load(task["config"]), completed)
        for r in records:
            r["analysis_stream"] = task["stream"]
        rows.extend(records)
        coverage.append(
            dict(
                index=task["index"],
                status=completed["status"],
                states=len(records) // 3,
            )
        )
    complete = len(rows) == 12 and all(
        t["status"] == "complete" and t["states"] == 1 for t in coverage
    )
    write_json(output / "coverage.json", dict(complete=complete, tasks=coverage))
    if not complete:
        write_json(
            output / "review.json",
            dict(
                status="incomplete_review_stop",
                coverage=coverage,
                original_gate2_unchanged=True,
                no_replacement=True,
            ),
        )
        archive()
        return
    manifest = methods()
    names = ["original_direct_seed42", "original_matched_h1_seed42"]
    forecasts = {}
    for name in names:
        spec = next(m for m in manifest["models"] if m["name"] == name)
        prediction = Forecaster.load(spec["folder"]).predict(rows)
        if not np.isfinite(prediction).all():
            raise RuntimeError("Nonfinite frozen transfer prediction")
        forecasts[name] = dict(
            predictions=prediction.tolist(),
            future_selection=future_selection(rows, prediction),
        )
    from diagnose_forecast_signal_v1 import groups, targets

    future = targets(rows, "future").mean(1)
    paired = []
    for group in groups(rows):
        values = []
        for name in names:
            p = np.array(forecasts[name]["predictions"])[group].mean(1).round(12)
            values.append(float(future[group][p == p.max()].mean()))
        paired.append(values[0] - values[1])
    write_json(
        output / "review.json",
        dict(
            status="external_four_state_review_required",
            states=4,
            trajectories=4,
            selection=selection(rows),
            frozen_forecast_transfer=forecasts,
            direct_h2_minus_matched_h1_future_value=stratified_interval(paired, rows),
            new_fits=0,
            original_gate2_unchanged=True,
            no_five_generation_third_model_claim=True,
            no_third_model_noise_floor_claim=True,
            limitation="Four early states; two seeds per task, shared evaluation sets and very limited inferential power",
        ),
    )
    archive()


def archive():
    import tarfile

    target = Path(
        "/work/11617/sujato_ts/vista/third_eye_results/smollm_supplement_v1.tar.gz"
    )
    if target.exists():
        raise RuntimeError("S1 durable archive already exists")
    partial = target.with_suffix(target.suffix + ".partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:
        tar.add(OUT, arcname=str(OUT.relative_to(ROOT)))
        for path in json.loads((OUT / "release.json").read_text())["files"]:
            tar.add(ROOT / path, arcname=path)
    partial.replace(target)
    write_json(
        OUT / "archive_receipt.json",
        dict(path=str(target), bytes=target.stat().st_size, sha256=file_digest(target)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["setup", "verify", "run", "review"])
    p.add_argument("index", nargs="?", type=int, choices=range(4))
    args = p.parse_args()
    if args.mode == "run":
        if args.index is None:
            p.error("Fixed single-state index required")
        run(args.index)
    else:
        {"setup": setup, "verify": verify, "review": review}[args.mode]()


if __name__ == "__main__":
    main()
