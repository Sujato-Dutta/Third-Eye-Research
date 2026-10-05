"""Audit a shorter retry cap from immutable A1 pools; never publish labels."""

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from third_eye.data.schema import Correction, Example
from third_eye.io import file_digest, write_json
from third_eye.updates.candidates import batch_hash, sample_batches


def corrections(items):
    return [
        Correction(Example(**x["example"]), x["completion"], x["attempt"])
        for x in items
    ]


def audit_pool(value, cap):
    stats, items = value["stats"], value["items"]
    original = stats["correction_attempt_limit"]
    if not isinstance(cap, int) or not 1 <= cap <= original:
        raise ValueError("Retry cap must be a positive prefix of the original budget")
    if stats["harvest_complete"] is not True:
        raise ValueError("Only complete historical harvests can be audited")
    ids = [x["example"]["id"] for x in items]
    if len(ids) != len(set(ids)) or stats["verified_corrections"] != len(items):
        raise ValueError("Invalid historical correction identities/count")
    if any(
        type(x["attempt"]) is not int or not 1 <= x["attempt"] <= original
        for x in items
    ):
        raise ValueError("Invalid first-success attempt")
    if any(
        x["example"]["split"] != "train" or not x["completion"].strip() for x in items
    ):
        raise ValueError("Require verified nonempty training corrections")
    unresolved = stats["unresolved_failures"]
    if unresolved < 0 or stats["initial_failures"] != unresolved + len(items):
        raise ValueError("Failure counts disagree with pool")
    actual_attempts = unresolved * original + sum(x["attempt"] for x in items)
    if stats["revision_attempts"] != actual_attempts:
        raise ValueError("Historical first-success stopping accounting disagrees")
    kept = [x for x in items if x["attempt"] <= cap]
    prefix_attempts = unresolved * cap + sum(min(cap, x["attempt"]) for x in items)
    return {
        "original_cap": original,
        "cap": cap,
        "original_verified": len(items),
        "retained_verified": len(kept),
        "lost_verified": len(items) - len(kept),
        "first_success_histogram": dict(
            sorted(Counter(x["attempt"] for x in items).items())
        ),
        "original_revision_attempts": actual_attempts,
        "prefix_revision_attempts": prefix_attempts,
        "saved_revision_attempts": actual_attempts - prefix_attempts,
        "original_difficulty": dict(Counter(x["example"]["difficulty"] for x in items)),
        "retained_difficulty": dict(Counter(x["example"]["difficulty"] for x in kept)),
    }, kept


def audit_plan(plan_path, cap):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = plan["stages"]["a1_pilots"]
    evidence = {str(plan_path): file_digest(plan_path)}
    states, incomplete = [], []
    for task in tasks:
        run = Path(task["output"])
        complete_path = run / "completed.json"
        if not complete_path.exists():
            incomplete.append({"stream": task["stream"], "seed": task["seed"]})
            continue
        completed = json.loads(complete_path.read_text())
        if completed["status"] != "complete" or completed["completed_states"] != 1:
            raise ValueError("Calibration run is not complete and one-state")
        label_path = run / "meta_labels.jsonl"
        rows = [json.loads(x) for x in label_path.read_text().splitlines() if x.strip()]
        if len(rows) != 3 or len({x["state_id"] for x in rows}) != 1:
            raise ValueError("Require a complete K=3 state")
        if any(x["protocol_amendment"] != "A1" for x in rows):
            raise ValueError("Require A1 provenance")
        evidence[str(label_path)] = file_digest(label_path)
        state_dir = run / "states" / rows[0]["state_id"]
        parent_path = state_dir / "correction_pool.json"
        parent_value = json.loads(parent_path.read_text())
        parent_audit, parent_kept = audit_pool(parent_value, cap)
        evidence[str(parent_path)] = file_digest(parent_path)
        seed = rows[0]["seed"] + rows[0]["generation"] * 1_000_000
        parent_ok = comb(len(parent_kept), 2) >= 3
        sampled = (
            sample_batches(corrections(parent_kept), 2, 3, seed, "stratified_distinct")
            if parent_ok
            else []
        )
        expected_hashes = [r["batch_hash"] for r in rows]
        original = sample_batches(
            corrections(parent_value["items"]), 2, 3, seed, "stratified_distinct"
        )
        if [batch_hash(x) for x in original] != expected_hashes:
            raise ValueError("Original candidate sampling does not replay")
        branches = []
        for row in rows:
            path = state_dir / row["candidate_id"] / "continuation_pool.json"
            value = json.loads(path.read_text())
            audit, kept = audit_pool(value, cap)
            if value["stats"] != row["continuation"]["pool_statistics"]:
                raise ValueError("Historical continuation pool disagrees with labels")
            evidence[str(path)] = file_digest(path)
            branches.append(
                {
                    "candidate_id": row["candidate_id"],
                    "pool": audit,
                    "prefix_continuation_available_on_original_checkpoint": int(
                        len(kept) >= 2
                    ),
                }
            )
        states.append(
            {
                "stream": task["stream"],
                "seed": task["seed"],
                "parent": parent_audit,
                "prefix_parent_can_form_k3": parent_ok,
                "candidate_batches_unchanged": [batch_hash(x) for x in sampled]
                == expected_hashes,
                "branches": branches,
            }
        )
    # Detect mutation of any input consumed by the audit.
    if any(file_digest(path) != fingerprint for path, fingerprint in evidence.items()):
        raise RuntimeError("Historical evidence changed during audit")
    summaries = {}
    for stream in sorted({s["stream"] for s in states}):
        group = [s for s in states if s["stream"] == stream]
        pools = [
            p for s in group for p in [s["parent"], *[b["pool"] for b in s["branches"]]]
        ]
        summaries[stream] = {
            "states": len(group),
            "parents_lacking_k3": sum(
                not s["prefix_parent_can_form_k3"] for s in group
            ),
            "states_with_changed_candidates": sum(
                not s["candidate_batches_unchanged"] for s in group
            ),
            "terminal_continuations_on_original_checkpoints": sum(
                not b["prefix_continuation_available_on_original_checkpoint"]
                for s in group
                for b in s["branches"]
            ),
            "original_verified": sum(p["original_verified"] for p in pools),
            "retained_verified": sum(p["retained_verified"] for p in pools),
            "original_revision_attempts": sum(
                p["original_revision_attempts"] for p in pools
            ),
            "prefix_revision_attempts": sum(
                p["prefix_revision_attempts"] for p in pools
            ),
        }
    return {
        "status": "complete",
        "purpose": "historical retry-prefix diagnostic; not research labels",
        "cap": cap,
        "states": states,
        "by_stream": summaries,
        "incomplete_runs": incomplete,
        "source_sha256": evidence,
        "scientific_runs_changed": False,
        "limitations": [
            "Preserves original per-problem draws with seed stride sixteen; setting correction_attempts=8 alone changes seeds.",
            "Counts saved generation attempts, not saved wall-clock time.",
            "Continuations are conditional on the original t+1 checkpoints; changed candidates need new H=2 experiments.",
            "Historical pool sufficiency cannot establish Gate 1/2 or recursive performance under the shorter protocol.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Diagnostic outputs are immutable")
    write_json(args.output, audit_plan(args.plan, args.cap))
    print(json.dumps({"output": str(args.output), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
