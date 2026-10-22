"""Outcome-blind exposure checks and fixed development timing samples."""

import json
from pathlib import Path
import re

from third_eye.data.benchmarks import canonical_prompt
from third_eye.io import digest, file_digest


DECLARATION = {
    "version": "scoring_precision_20261021",
    "purpose": "development_inference_cost_and_recorded_item_exposure_only",
    "automatic_expansion": False,
    "benchmark_seed": 12142,
    "candidate_row": 0,
    "continuation_draw": 0,
    "development_items_per_role": 16,
    "checkpoints": ["parent", "first_candidate_first_continuation"],
    "selection": "evenly spaced prompt-character-length ranks; id breaks ties",
    "final_predictions_generated": False,
    "final_outcomes_extracted_or_used": False,
    "max_reservation_sus": 8.605,
    "limitations": "Recorded exposure is scoped to searched artifacts, not proof of no historical or pretraining exposure. Timing uses development items, not final outcomes. Code final OOD changes benchmark and remains unpriced directly.",
}


def timing_sample(examples, size=16):
    """Include both length extremes without inspecting answers or outcomes."""
    if size < 2 or len(examples) < size:
        raise ValueError("Enough distinct development items required")
    if any(not ex.split.endswith("_dev") for ex in examples):
        raise ValueError("Timing must not generate final or training outcomes")
    if len({ex.id for ex in examples}) != len(examples):
        raise ValueError("Timing items must be unique")
    ordered = sorted(examples, key=lambda ex: (len(ex.prompt), ex.id))
    return [ordered[i * (len(ordered) - 1) // (size - 1)] for i in range(size)]


def identity_index(suites):
    """All matches are conservative; raw source identity survives wrappers."""
    index = {}
    for suite, roles in suites.items():
        for role, examples in roles.items():
            for ex in examples:
                key = (suite, role, ex.id)
                for identity in (
                    ("id", ex.id),
                    ("prompt", ex.prompt_hash),
                    ("source", digest(canonical_prompt(ex))),
                ):
                    index.setdefault(identity, set()).add(key)
    return index


def record_matches(value, index, suite_hashes):
    """Extract identities only. Never use correctness/completion content."""
    matched, opaque_final = set(), False

    def visit(node):
        nonlocal opaque_final
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        # Suite references alone are planned access, not evidence of evaluation.
        evaluated_suite = "scores" in node and (
            "evaluation" in node or "final_manifest_hash" in node
        )
        if evaluated_suite:
            suite = suite_hashes.get(node.get("final_manifest_hash"))
            if suite is not None:
                matched.update(suite)
            elif "final_manifest_hash" in node:
                opaque_final = True
        observed = any(
            k in node for k in ("correct", "passed", "prediction_sha256", "completion")
        )
        if (
            observed
            and str(node.get("role", node.get("split", ""))).endswith("_test")
            and not node.get("id")
        ):
            opaque_final = True
        # Full examples embedded in correction batches also count as exposure.
        used_example = {"id", "prompt", "answer", "split"}.issubset(node)
        if (observed or used_example) and isinstance(node.get("id"), str):
            matched.update(index.get(("id", node["id"]), ()))
            if isinstance(node.get("prompt_sha256"), str):
                matched.update(index.get(("prompt", node["prompt_sha256"]), ()))
            if isinstance(node.get("prompt"), str):
                normalized = " ".join(node["prompt"].split()).casefold()
                matched.update(index.get(("prompt", digest(normalized)), ()))
                source = node.get("metadata", {}).get("source_prompt", node["prompt"])
                matched.update(
                    index.get(
                        ("source", digest(" ".join(source.split()).casefold())), ()
                    )
                )
        for child in node.values():
            if isinstance(child, (list, dict)):
                visit(child)

    visit(value)
    return matched, opaque_final


def scan_exposure(paths, index, suite_hashes):
    """Fail closed on unreadable artifacts; only matched identities leave scan."""
    hits, opaque, errors, inventory = {}, [], [], []
    for name in sorted({str(Path(p)) for p in paths}):
        path = Path(name)
        try:
            inventory.append({"path": name, "sha256": file_digest(path)})
            with path.open(encoding="utf-8") as stream:
                records = (
                    (json.loads(line) for line in stream if line.strip())
                    if path.suffix == ".jsonl"
                    else (json.load(stream),)
                )
                for record in records:
                    matches, unknown = record_matches(record, index, suite_hashes)
                    for key in matches:
                        hits.setdefault(key, set()).add(name)
                    if unknown:
                        opaque.append(name)
        except (OSError, ValueError, TypeError, AttributeError, RecursionError) as exc:
            errors.append({"path": name, "error": type(exc).__name__})
    return hits, sorted(set(opaque)), errors, inventory


def benchmark_checkpoints(row):
    """The native draw zero is the recorded t2, never an inferred pathname."""
    draw = row["continuation_replicates"][0]
    parent_path = Path(row["checkpoints"]["t1"]).parent.parent / "parent"
    return [
        ("parent", parent_path, row["parent_adapter_hash"]),
        ("future", Path(row["checkpoints"]["t2"]), draw["continuation_adapter_hash"]),
    ]


def synthetic_fixture_records(path, records):
    """Require known fixture provenance AND mock content; paths alone never suffice."""
    name = Path(path).as_posix()
    if not re.search(r"/(?:cpu_)?validation/fixtures/test_", name) or not records:
        return False
    if re.search(
        r"/test_result_cli_separates_fina\d+/(?:direct|greedy_h1)_4[234]/final/scores\.json$",
        name,
    ):
        return len(records) == 1 and all(
            records[0].get(key) == expected
            for key, expected in {
                "model": "fixture-model",
                "final_manifest_hash": "fixture-final",
                "config_hash": "cfg",
                "resolved_revision": "a" * 40,
            }.items()
        )
    if not re.search(
        r"/test_prepare_portable_suites_a\d+/(?:original|relocated)/[a-z_]+\.jsonl$",
        name,
    ):
        return False
    for row in records:
        metadata = row.get("metadata", {})
        repository = metadata.get("source_repository")
        source = metadata.get("source_prompt", "")
        if (
            metadata.get("source_revision") != "a" * 40
            or repository not in {"openai/gsm8k", "google-research-datasets/mbpp"}
            or not source.startswith(repository + " ")
            or not re.search(r" distinct (?:problem|code) \d+$", source)
            or not row.get("prompt", "").startswith(source)
        ):
            return False
    return True
