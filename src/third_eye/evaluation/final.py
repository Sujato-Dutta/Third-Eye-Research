"""Final evaluation of sealed checkpoints; never supplies selection features."""

import json
from pathlib import Path

from third_eye.data.schema import TEST_ROLES
from third_eye.data.splits import load_manifest, read_examples
from third_eye.data.benchmarks import audit_overlap
from third_eye.io import digest, file_digest, append_jsonl, write_json
from third_eye.provenance import source_inventory, versions, execution_environment


def load_final_manifest(path, selection_path):
    path = Path(path)
    final = json.loads(path.read_text(encoding="utf-8"))
    selection, splits = load_manifest(selection_path)
    if final.get("schema_version") != 1 or set(final.get("splits", {})) != set(
        TEST_ROLES
    ):
        raise ValueError("Final evaluation requires exactly three final test splits")
    if final.get("selection_hash") != digest(selection):
        raise ValueError("Final suite does not match the audited development manifest")
    for role, entry in final["splits"].items():
        source = path.parent / entry["path"]
        if file_digest(source) != entry["sha256"]:
            raise ValueError("Final benchmark checksum mismatch")
        splits[role] = read_examples(source, role)
        if len(splits[role]) != entry["count"]:
            raise ValueError("Final benchmark count mismatch")
    audit_overlap(splits)
    return final, {role: splits[role] for role in TEST_ROLES}


def evaluate_final(backend, splits, verifier, output, max_new_tokens, seed, metadata):
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Final evaluation output is immutable; choose a new directory"
        )
    if set(splits) != set(TEST_ROLES):
        raise ValueError("Only sealed final suites may enter final evaluation")
    output.mkdir(parents=True)
    before = backend.state_hash()
    scores = {}
    from third_eye.evaluation.runner import generate_greedy, verify_completions

    for role, examples in splits.items():
        count = 0
        completions = generate_greedy(
            backend, examples, max_new_tokens, [seed] * len(examples)
        )
        verdicts = verify_completions(verifier, examples, completions)
        for example, completion, passed in zip(examples, completions, verdicts):
            count += passed
            append_jsonl(
                output / "items.jsonl",
                {
                    "id": example.id,
                    "role": role,
                    "passed": passed,
                    "completion": completion,
                    "benchmark": example.metadata.get("benchmark"),
                },
            )
        scores[role] = count / len(examples)
    if backend.state_hash() != before:
        raise RuntimeError("Final evaluation mutated model parameters")
    result = {
        "schema_version": 1,
        "scores": scores,
        "counts": {k: len(v) for k, v in splits.items()},
        "adapter_hash": before,
        "seed": seed,
        "evaluation": "greedy pass@1 with conditional multiple-choice scoring"
        if getattr(
            getattr(getattr(backend, "config", None), "protocol", None),
            "multiple_choice_scoring",
            "generation",
        )
        == "conditional_likelihood"
        else "greedy pass@1",
        "software_versions": versions(),
        "execution": execution_environment(),
        **source_inventory(),
        **metadata,
    }
    write_json(output / "scores.json", result)
    return result
