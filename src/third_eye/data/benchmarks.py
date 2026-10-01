"""Pinned benchmark preparation with independent selection and final suites."""

from dataclasses import asdict
from pathlib import Path
import random
import re

from third_eye.data.schema import Example, ROLES, TEST_ROLES
from third_eye.data.splits import create_manifest
from third_eye.io import digest, file_digest, publish_jsonl_batch, write_json

MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def boxed_answer(solution):
    """Extract the last balanced LaTeX box, including nested fractions."""
    start = solution.rfind("\\boxed{")
    if start < 0:
        raise ValueError("MATH reference has no boxed answer")
    start += len("\\boxed{")
    depth = 1
    for i in range(start, len(solution)):
        depth += (solution[i] == "{") - (solution[i] == "}")
        if depth == 0:
            return solution[start:i]
    raise ValueError("Unbalanced MATH reference answer")


def canonical_prompt(example):
    return " ".join(
        example.metadata.get("source_prompt", example.prompt).split()
    ).casefold()


def audit_overlap(suites):
    """Check raw prompts as well as IDs; instructions cannot hide overlap.

    Also flag near duplicates by token shingles. Near matches are excluded from
    development/update data by preparation, preserving final benchmark sets.
    """
    ids, prompts = {}, {}
    for role, examples in suites.items():
        for ex in examples:
            prompt = digest(canonical_prompt(ex))
            if ex.id in ids or prompt in prompts:
                raise ValueError(
                    f"Duplicate benchmark source across {role} and another split: {ex.id}"
                )
            ids[ex.id], prompts[prompt] = role, role


def deduplicate(suites, threshold=0.85):
    """Final suites have priority; remove overlapping development examples."""
    accepted, seen_ids, seen_prompts, shingles, inverted, removed = (
        {},
        set(),
        set(),
        [],
        {},
        [],
    )
    order = list(TEST_ROLES) + ["retention_dev", "ood_dev", "target_dev", "train"]
    for role in order:
        accepted[role] = []
        for ex in suites[role]:
            text = canonical_prompt(ex)
            tokens = re.findall(r"\w+", text)
            grams = {tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)}
            nearby = (
                set().union(*(inverted.get(g, set()) for g in grams))
                if grams
                else set()
            )
            near = any(
                len(grams & shingles[i]) / len(grams | shingles[i]) >= threshold
                for i in nearby
            )
            duplicate = (
                ex.id in seen_ids
                or text in seen_prompts
                or (near and role not in TEST_ROLES)
            )
            if duplicate:
                removed.append({"id": ex.id, "role": role, "near_duplicate": near})
                continue
            accepted[role].append(ex)
            seen_ids.add(ex.id)
            seen_prompts.add(text)
            index = len(shingles)
            shingles.append(grams)
            for gram in grams:
                inverted.setdefault(gram, set()).add(index)
    audit_overlap(accepted)
    return accepted, removed


def prepare(output, task, seed=42, train_size=256, dev_size=64, retention_size=256):
    """Resolve immutable Hub revisions and archive all preparation choices.

    MATH-500 and HumanEval are final-only. Code development uses a separate
    MBPP validation proxy; its score must not be described as HumanEval OOD.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Benchmark versions are immutable; choose a new output directory"
        )
    if task not in {"math", "code"}:
        raise ValueError("Unknown task family")
    revisions = {}

    def load(repo, config, split):
        if repo not in revisions:
            revisions[repo] = HfApi().dataset_info(repo).sha
        data = load_dataset(repo, config, split=split, revision=revisions[repo])
        return [
            dict(row, _index=i, _repo=repo, _source_split=split)
            for i, row in enumerate(data)
        ]

    def shuffle(rows, salt=0):
        rows = list(rows)
        random.Random(seed + salt).shuffle(rows)
        return rows

    def convert(rows, role, benchmark):
        result = []
        for row in rows:
            meta = {
                "benchmark": benchmark,
                "source_repository": row["_repo"],
                "source_revision": revisions[row["_repo"]],
                "source_split": row["_source_split"],
            }
            identifier = str(row.get("task_id", row["_index"]))
            if benchmark == "gsm8k":
                prompt, answer, kind = row["question"], row["answer"], "math"
                instruction = "\nSolve step by step. End with #### followed by the numeric answer."
            elif benchmark in {"math", "math500"}:
                prompt = row["problem"]
                answer = (
                    row["answer"] if "answer" in row else boxed_answer(row["solution"])
                )
                kind, instruction = (
                    "math",
                    "\nSolve step by step and put the final answer in \\boxed{}.",
                )
                meta["verifier"] = "symbolic_math"
                meta["subject"] = row.get("type", row.get("subject", "unknown"))
                identifier = meta["subject"] + ":" + identifier
            elif benchmark == "mbpp":
                prompt, answer, kind = row["text"], row["code"], "code"
                import ast

                tree = ast.parse(answer)
                signatures = [
                    f"{node.name}({ast.unparse(node.args)})"
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                instruction = "\nReturn Python code only. Implement: " + ", ".join(
                    signatures
                )
                meta.update(
                    test_list=row["test_list"],
                    test_setup_code=row.get("test_setup_code", ""),
                )
            elif benchmark == "humaneval":
                prompt, answer, kind = row["prompt"], row["canonical_solution"], "code"
                instruction = (
                    "\nReturn the complete Python function, preserving its signature."
                )
                meta.update(
                    test=row["test"], entry_point=row["entry_point"], code_prefix=prompt
                )
            else:
                prompt = (
                    row["question"]
                    + "\n"
                    + "\n".join(
                        f"{chr(65 + i)}. {c}" for i, c in enumerate(row["choices"])
                    )
                )
                answer, kind, instruction = (
                    chr(65 + row["answer"]),
                    "exact",
                    "\nReturn only the answer letter A, B, C, or D.",
                )
                meta["verifier"] = "multiple_choice"
                identifier = row.get("subject", "all") + ":" + identifier
            meta["source_prompt"] = prompt
            # MBPP task IDs are global across splits; other datasets use split-local IDs.
            source_split = "" if benchmark == "mbpp" else row["_source_split"] + ":"
            difficulty = str(
                row.get("level", "long" if len(prompt.split()) >= 60 else "short")
            )
            result.append(
                Example(
                    f"{benchmark}:{source_split}{identifier}",
                    prompt + instruction,
                    answer,
                    role,
                    kind,
                    difficulty,
                    meta,
                )
            )
        return result

    suites = {}
    if task == "math":
        pool = shuffle(load("openai/gsm8k", "main", "train"))
        suites["target_dev"] = convert(pool[:dev_size], "target_dev", "gsm8k")
        suites["train"] = convert(
            pool[dev_size : dev_size + train_size], "train", "gsm8k"
        )
        math_pool = []
        for subject in MATH_SUBJECTS:
            math_pool.extend(load("EleutherAI/hendrycks_math", subject, "train"))
        suites["ood_dev"] = convert(shuffle(math_pool, 1)[:dev_size], "ood_dev", "math")
        suites["target_test"] = convert(
            load("openai/gsm8k", "main", "test"), "target_test", "gsm8k"
        )
        suites["ood_test"] = convert(
            load("HuggingFaceH4/MATH-500", None, "test"), "ood_test", "math500"
        )
    else:
        pool = shuffle(load("google-research-datasets/mbpp", "full", "train"))
        suites["target_dev"] = convert(pool[:dev_size], "target_dev", "mbpp")
        suites["train"] = convert(
            pool[dev_size : dev_size + train_size], "train", "mbpp"
        )
        suites["ood_dev"] = convert(
            shuffle(load("google-research-datasets/mbpp", "full", "validation"), 1)[
                :dev_size
            ],
            "ood_dev",
            "mbpp",
        )
        suites["target_test"] = convert(
            load("google-research-datasets/mbpp", "full", "test"), "target_test", "mbpp"
        )
        suites["ood_test"] = convert(
            load("openai/openai_humaneval", None, "test"), "ood_test", "humaneval"
        )
    suites["retention_dev"] = convert(
        shuffle(load("cais/mmlu", "all", "validation"), 2)[:retention_size],
        "retention_dev",
        "mmlu",
    )
    suites["retention_test"] = convert(
        shuffle(load("cais/mmlu", "all", "test"), 3)[:retention_size],
        "retention_test",
        "mmlu",
    )
    suites, removed = deduplicate(suites)
    for role, rows in suites.items():
        if not rows:
            raise ValueError(f"Empty {role} after overlap audit")
    output.mkdir(parents=True)
    for role, rows in suites.items():
        publish_jsonl_batch(output / f"{role}.jsonl", [asdict(ex) for ex in rows])
    selection = create_manifest(
        {role: output / f"{role}.jsonl" for role in ROLES}, output / "selection.json"
    )
    write_json(
        output / "final.json",
        {
            "schema_version": 1,
            "selection_hash": digest(selection),
            "splits": {
                role: {
                    "path": f"{role}.jsonl",
                    "sha256": file_digest(output / f"{role}.jsonl"),
                    "count": len(suites[role]),
                }
                for role in TEST_ROLES
            },
        },
    )
    write_json(
        output / "provenance.json",
        {
            "seed": seed,
            "task": task,
            "revisions": revisions,
            "requested_sizes": {
                "train": train_size,
                "dev": dev_size,
                "retention": retention_size,
            },
            "counts": {role: len(rows) for role, rows in suites.items()},
            "excluded_overlap": removed,
            "code_development_proxy": "MBPP validation; not HumanEval"
            if task == "code"
            else None,
            "audit": "source IDs, normalized raw prompts, five-token shingle Jaccard >=0.85",
        },
    )
    return output / "selection.json"
