"""Allocated export of immutable confirmation inputs for paired DGX controls."""

import argparse
import json
from pathlib import Path
import tarfile

import numpy as np

from forecasting_pilots import inspect_task
from third_eye.cluster import require_compute_allocation
from third_eye.config import Config
from third_eye.data.splits import load_manifest
from third_eye.evaluation.final import load_final_manifest
from third_eye.experiments.confirmation import actions, checkpoints, read_scored_items
from third_eye.io import digest, file_digest, write_json
from third_eye.training.sources import resolve_source
from third_eye.updates.candidates import batch_hash
from third_eye.data.schema import Correction, Example


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_compute_allocation()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(exist_ok=False)
    subset = read(args.subset)
    core = root / "runs/final_empirical_confirmation_43b5c215c6b0"
    if file_digest(core / "plan.json") != subset["source_cohort_plan_sha256"]:
        raise ValueError("Original confirmation plan differs")
    plan = read(core / "plan.json")
    selected = {(t["stream"], t["seed"]) for t in subset["existing_parent_tasks"]}
    files, models, tasks, states = {}, {}, [], []

    def bind(path):
        path = Path(path)
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "path": str(path),
            "sha256": file_digest(path),
            "bytes": path.stat().st_size,
        }
        return relative

    def checkpoint(path, expected):
        path = Path(path)
        saved = read(path / "state.json")
        if saved["adapter_hash"] != expected:
            raise ValueError("Adapter identity differs")
        bind(path / "state.json")
        bind(path / "adapter_config.json")
        for name, sha in saved["weights"].items():
            if file_digest(path / name) != sha:
                raise ValueError("Adapter weight checksum differs")
            bind(path / name)
        return path.relative_to(root).as_posix()

    for task in plan["tasks"]:
        folder = Path(task["output"])
        done, decision = (
            read(folder / "completed.json"),
            read(folder / "commitment.json"),
        )
        if not done["passed"] or done["commitment_sha256"] != file_digest(
            folder / "commitment.json"
        ):
            raise ValueError("Immutable complete parent required")
        if (
            actions(task["stream"], decision["development_h1_heads"])
            != decision["actions"]
        ):
            raise ValueError("Development-only selection differs")
        rows, _ = inspect_task(
            {**task, "output": str(folder / "collection")},
            [50_000_000, 100_000_000],
            plan["source_tree_sha256"],
        )
        _, final = load_final_manifest(task["final_manifest"], task["manifest"])
        artifacts = {}
        for name, _, adapter_hash in checkpoints(rows):
            dest = folder / "scoring" / name
            for filename, key in [
                ("scores.json", "scores_sha256"),
                ("items.jsonl", "items_sha256"),
            ]:
                if file_digest(dest / filename) != done["scores"][name][key]:
                    raise ValueError("Original item outcomes changed")
                bind(dest / filename)
            artifacts[name] = {
                r: x.tolist()
                for r, x in read_scored_items(
                    dest, final, adapter_hash, done["commitment_sha256"]
                ).items()
            }
        states.append(
            {
                "stream": task["stream"],
                "seed": task["seed"],
                "actions": decision["actions"],
                "artifacts": artifacts,
                "ids": {
                    r: [(e.id, e.prompt_hash) for e in examples]
                    for r, examples in final.items()
                },
            }
        )
        if (task["stream"], task["seed"]) not in selected:
            continue
        key = [subset["selection_namespace"], task["stream"], task["seed"]]
        eligible = np.flatnonzero(np.asarray(decision["actions"]["forced"]) > 0)
        k = min(eligible, key=lambda i: digest([*key, "candidate", int(i)]))
        row = rows[k]
        branch = Path(row["checkpoints"]["t1"]).parent
        cfg = Config.load(task["config"])
        manifests = {}
        for field, loader in [
            ("manifest", load_manifest),
            ("final_manifest", lambda p: load_final_manifest(p, task["manifest"])),
        ]:
            meta, _ = loader(task[field])
            manifests[field] = bind(task[field])
            for entry in meta["splits"].values():
                bind(Path(task[field]).parent / entry["path"])
        draws = []
        for d, obs in enumerate(row["continuation_replicates"]):
            pool_path = (
                branch / "continuation_pool.json"
                if d == 0
                else branch / "continuation_replicates" / str(obs["seed"]) / "pool.json"
            )
            batch_path = (
                branch / "continuation_batch.json"
                if d == 0
                else pool_path.with_name("batch.json")
            )
            batch = read(batch_path)
            pool = read(pool_path)
            parsed = [
                Correction(Example(**c["example"]), c["completion"], c["attempt"])
                for c in batch
            ]
            if (batch_hash(parsed) if parsed else None) != obs.get("batch_hash", row["continuation"]["batch_hash"]):
                raise ValueError("Saved verified continuation batch differs")
            if any(digest(c) not in {digest(v) for v in pool["items"]} for c in batch):
                raise ValueError("Correction absent from verified pool")
            draws.append(
                {
                    "index": d,
                    "observation": obs,
                    "batch": batch,
                    "batch_file": bind(batch_path),
                    "pool_file": bind(pool_path),
                }
            )
        chosen = sorted(
            range(3),
            key=lambda d: digest([*key, "draw", draws[d]["observation"]["seed"]]),
        )[:2]
        parent_path = Path(row["checkpoints"]["t1"]).parent.parent / "parent"
        tasks.append(
            {
                "stream": task["stream"],
                "seed": task["seed"],
                "config": bind(task["config"]),
                **manifests,
                "candidate_index": int(k),
                "tie_rule": "minimum frozen SHA256 among development-optimal candidates",
                "selected_draws": chosen,
                "draws": draws,
                "m0": checkpoint(parent_path, row["parent_adapter_hash"]),
                "m1": checkpoint(
                    row["checkpoints"]["t1"], row["candidate_adapter_hash"]
                ),
                "candidate_hash": row["candidate_adapter_hash"],
                "parent_hash": row["parent_adapter_hash"],
                "new_harvest_seeds": {
                    str(d): 1_500_000_000 + task["index"] * 100_000 + d * 10_000
                    for d in chosen
                },
            }
        )
        if cfg.model.name not in models:
            source, revision, proof = resolve_source(cfg.model)
            if Path(source).is_dir():
                snapshot = Path(source)
            else:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(
                    source,
                    "config.json",
                    revision=revision,
                    cache_dir=root / "models/hf_cache/hub",
                )
                if not isinstance(cached, str):
                    raise ValueError("Original cached backbone metadata required")
                snapshot = Path(cached).parent
            model_files = {}
            for p in sorted(snapshot.iterdir()):
                if p.is_file() and p.suffix in {
                    ".json",
                    ".jinja",
                    ".txt",
                    ".model",
                    ".safetensors",
                }:
                    model_files[p.name] = {
                        "sha256": file_digest(p),
                        "bytes": p.stat().st_size,
                        "original_path": str(p),
                    }
            models[cfg.model.name] = {
                "model": cfg.model.name,
                "revision": cfg.model.revision,
                "source_repository": proof["source_repository"] if proof else source,
                "source_revision": revision,
                "source_proof": proof,
                "files": model_files,
            }
    if len(states) != 32 or len(tasks) != 16 or len(models) != 2:
        raise ValueError(
            "Complete original cohort and frozen sixteen-parent subset required"
        )
    registry = {
        "original_root": str(root),
        "original_plan_sha256": file_digest(core / "plan.json"),
        "subset_sha256": file_digest(args.subset),
        "tasks": tasks,
        "states": states,
        "models": models,
        "files": files,
        "new_data_selection_uses_final_outcomes": False,
        "cross_hardware": "All new contrasts require a fresh native 50-step baseline on DGX; original Vista outcomes remain separate",
    }
    write_json(output / "registry.json", registry)
    with tarfile.open(output / "payload.tar.gz", "w:gz", dereference=True) as archive:
        archive.add(output / "registry.json", arcname="registry.json")
        for relative, item in files.items():
            archive.add(item["path"], arcname="payload/" + relative, recursive=False)
    write_json(
        output / "completed.json",
        {
            "passed": True,
            "parents": 16,
            "states": 32,
            "registry_sha256": file_digest(output / "registry.json"),
            "archive_sha256": file_digest(output / "payload.tar.gz"),
            "archive_bytes": (output / "payload.tar.gz").stat().st_size,
        },
    )
    print(json.dumps(read(output / "completed.json")))


if __name__ == "__main__":
    main()
