"""Expand only A1 training harvests at original dataset revisions on CPU nodes."""

import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def convert_training(rows, task, revision):
    from third_eye.data.schema import Example

    result = []
    for index, row in rows:
        if task == "math":
            prompt, answer = row["question"], row["answer"]
            instruction = (
                "\nSolve step by step. End with #### followed by the numeric answer."
            )
            identifier = f"gsm8k:train:{index}"
            metadata = {"benchmark": "gsm8k", "source_repository": "openai/gsm8k"}
        else:
            prompt, answer = row["text"], row["code"]
            signatures = [
                f"{n.name}({ast.unparse(n.args)})"
                for n in ast.parse(answer).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            instruction = "\nReturn Python code only. Implement: " + ", ".join(
                signatures
            )
            identifier = "mbpp:" + str(row["task_id"])
            metadata = {
                "benchmark": "mbpp",
                "source_repository": "google-research-datasets/mbpp",
                "test_list": row["test_list"],
                "test_setup_code": row.get("test_setup_code", ""),
            }
        metadata.update(
            source_revision=revision, source_split="train", source_prompt=prompt
        )
        result.append(
            Example(
                identifier,
                prompt + instruction,
                answer,
                "train",
                task,
                "long" if len(prompt.split()) >= 60 else "short",
                metadata,
            )
        )
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("A1 preparation requires a compute allocation")
    from datasets import load_dataset
    from third_eye.data.benchmarks import deduplicate
    from third_eye.data.schema import ROLES, TEST_ROLES
    from third_eye.data.splits import load_manifest, read_examples, create_manifest
    from third_eye.io import digest, file_digest, publish_jsonl_batch, write_json
    from third_eye.provenance import source_inventory, versions

    original = Path(os.environ["THIRD_EYE_ORIGINAL_ROOT"])
    counts = {}
    for task in ("math", "code"):
        old = original / "data/processed/v1" / task
        provenance = json.loads((old / "provenance.json").read_text())
        manifest, suites = load_manifest(old / "selection.json")
        repo = "openai/gsm8k" if task == "math" else "google-research-datasets/mbpp"
        revision = provenance["revisions"][repo]
        rows = list(
            enumerate(
                load_dataset(
                    repo,
                    "main" if task == "math" else "full",
                    split="train",
                    revision=revision,
                )
            )
        )
        random.Random(42).shuffle(rows)
        selected = rows[64:1088] if task == "math" else rows
        suites["train"] = convert_training(selected, task, revision)
        # Final examples are used solely for the pre-existing prompt-overlap
        # audit; no model evaluation or final-score selection occurs here.
        for role in TEST_ROLES:
            suites[role] = read_examples(old / (role + ".jsonl"), role)
        suites, removed = deduplicate(suites)
        out = ROOT / "data/processed/a1" / task
        out.mkdir(parents=True, exist_ok=False)
        publish_jsonl_batch(out / "train.jsonl", [asdict(ex) for ex in suites["train"]])
        for role in (*ROLES[1:], *TEST_ROLES):
            shutil.copyfile(old / (role + ".jsonl"), out / (role + ".jsonl"))
            if file_digest(out / (role + ".jsonl")) != file_digest(
                old / (role + ".jsonl")
            ):
                raise RuntimeError("Evaluation file changed during A1 preparation")
        selection = create_manifest(
            {r: out / (r + ".jsonl") for r in ROLES}, out / "selection.json"
        )
        final = json.loads((old / "final.json").read_text())
        final["selection_hash"] = digest(selection)
        write_json(out / "final.json", final)
        write_json(
            out / "provenance.json",
            {
                **provenance,
                "amendment": "A1",
                "original_manifest_sha256": file_digest(old / "selection.json"),
                "harvest_source": repo,
                "harvest_source_revision": revision,
                "predeclared_portion": "seed42 ranks 64:1088"
                if task == "math"
                else "all official train rows, seed42 order",
                "harvest_selected_before_audit": len(selected),
                "harvest_count": len(suites["train"]),
                "a1_excluded_overlap": removed,
                "evaluation_files_unchanged": True,
            },
        )
        counts[task] = len(suites["train"])
    write_json(
        ROOT / "runs/a1_cpu_passed.json",
        {
            "status": "passed",
            "job_id": os.environ["SLURM_JOB_ID"],
            "training_counts": counts,
            "amendment_sha256": file_digest(ROOT / "docs/protocol_amendment_a1.md"),
            "software_versions": versions(),
            **source_inventory(ROOT),
        },
    )
    print(json.dumps({"status": "passed", "training_counts": counts}))


if __name__ == "__main__":
    main()
