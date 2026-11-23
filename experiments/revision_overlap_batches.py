"""Build training-only batches with a controlled arithmetic-profile difference."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import numpy as np

from third_eye.cluster import require_compute_allocation
from third_eye.data.schema import Correction, Example
from third_eye.io import digest, file_digest, write_json
from third_eye.training.tokenization import encode_completion, frozen_chat_kwargs

CALCULATION = re.compile(r"<<([^=<>]+)=[^<>]+>>")
OPERATORS = re.compile(r"(?<=[\d)])\s*([+\-*/])(?=\s*[-+(\d])")


def profile(reference):
    """Use GSM8K calculator annotations as a training-only skill proxy."""
    formulas = CALCULATION.findall(reference)
    operations = Counter(op for formula in formulas for op in OPERATORS.findall(formula))
    counts = np.asarray([operations[op] for op in "+-*/"], dtype=float)
    if counts.sum() == 0:
        return None
    return counts / counts.sum(), min(len(formulas), 4)


def matched_batches(first, pool, token_counts, seed, size=8, shared_doses=False):
    """Match each dose within annotated-step bins; retain scarcity explicitly."""
    if size < 1 or not first:
        raise ValueError("Expected a nonempty first batch and a positive dose size")
    first_ids, profiles = set(), []
    for correction in first:
        first_ids.add(correction.example.id)
        tagged = profile(correction.example.answer)
        if tagged:
            profiles.append(tagged[0])
    if len(first_ids) != len(first) or not profiles:
        return {"available": False, "reason": "unusable first-batch arithmetic profile"}
    center = np.mean(profiles, axis=0)
    rng = random.Random(seed)
    groups = [[] for _ in range(4)]
    tagged_count = 0
    seen = set()
    for correction in sorted(pool, key=lambda c: c.example.id):
        ex = correction.example
        if ex.id in seen:
            raise ValueError("Correction pool repeats a problem")
        seen.add(ex.id)
        tagged = profile(ex.answer)
        if ex.id in first_ids or tagged is None:
            continue
        vector, steps = tagged
        similarity = float(vector @ center / (np.linalg.norm(vector)*np.linalg.norm(center)))
        groups[steps-1].append((similarity, rng.random(), token_counts[ex.id], correction))
        tagged_count += 1
    if shared_doses:
        return compare_batched_doses(groups, tagged_count, size, rng)
    triples = []
    for group in groups:
        ranked = sorted(group, key=lambda row: row[:2])
        width = len(ranked)//3
        if width == 0:
            continue
        thirds = [ranked[:width], ranked[width:-width], ranked[-width:]]
        # Pair by label-token length within the same difficulty-proxy bin.
        thirds = [sorted(rows, key=lambda row: (row[2], row[1])) for rows in thirds]
        for low, mid, high in zip(*thirds):
            span = high[0]-low[0]
            spread = max(low[2], mid[2], high[2])-min(low[2], mid[2], high[2])
            if span >= .05 and spread <= max(16, 0.25*min(low[2], mid[2], high[2])):
                triples.append((low, mid, high))
    if len(triples) < size:
        return {"available": False, "reason": "matched donor scarcity",
                "tagged_pool_size": tagged_count, "matched_triples": len(triples)}
    selected = rng.sample(triples, size)
    batches = {
        label: {"items": tuple(triple[i][3] for triple in selected),
                "mean_similarity": float(np.mean([triple[i][0] for triple in selected])),
                "label_tokens": sum(triple[i][2] for triple in selected)}
        for i, label in enumerate(("low", "medium", "high"))
    }
    if not batches["low"]["mean_similarity"] < batches["medium"]["mean_similarity"] < batches["high"]["mean_similarity"]:
        return {"available": False, "reason": "insufficient distinct dose means",
                "tagged_pool_size": tagged_count, "matched_triples": len(triples)}
    return {"available": True, "tagged_pool_size": tagged_count,
            "matched_triples": len(triples), "batches": batches}


def compare_batched_doses(groups, tagged_count, size, rng):
    """Allow shared examples across doses and match the whole-batch budgets."""
    if tagged_count < size + 2:
        return {"available": False, "reason": "too few distinct training examples",
                "tagged_pool_size": tagged_count}
    strata = [i for i, items in enumerate(groups) for _ in items]
    required = Counter(rng.sample(strata, size))
    samples = {}
    for _ in range(512):
        rows = tuple(row for i, count in required.items() for row in rng.sample(groups[i], count))
        key = tuple(sorted(row[3].example.id for row in rows))
        if key not in samples:
            samples[key] = {
                "items": tuple(row[3] for row in rows),
                "mean_similarity": sum(row[0] for row in rows) / size,
                "label_tokens": sum(row[2] for row in rows),
            }
    center = float(np.median([batch["label_tokens"] for batch in samples.values()]))
    matched = sorted(
        (batch for batch in samples.values() if abs(batch["label_tokens"]-center) <= .05*center),
        key=lambda batch: batch["mean_similarity"],
    )
    separation = matched[-1]["mean_similarity"] - matched[0]["mean_similarity"] if matched else 0
    details = {"tagged_pool_size": tagged_count, "budget_matched_candidates": len(matched),
               "max_mean_similarity_gap": separation}
    if len(matched) < 3 or separation < .05:
        return {"available": False, "reason": "insufficient distinct matched doses", **details}
    third = separation / 3
    low = [batch for batch in matched if batch["mean_similarity"] < matched[0]["mean_similarity"] + third]
    high = [batch for batch in matched if batch["mean_similarity"] > matched[-1]["mean_similarity"] - third]
    medium = [batch for batch in matched if
              matched[0]["mean_similarity"] + third <= batch["mean_similarity"] <= matched[-1]["mean_similarity"] - third]
    if not medium:
        return {"available": False, "reason": "no budget-matched medium dose", **details}
    low = [batch for batch in low if high[-1]["mean_similarity"] - batch["mean_similarity"] >= .05]
    chosen_low = rng.choice(low)
    high = [batch for batch in high if batch["mean_similarity"] - chosen_low["mean_similarity"] >= .05]
    return {"available": True, **details,
            "batches": {"low": chosen_low, "medium": rng.choice(medium), "high": rng.choice(high)}}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def corrections(rows):
    return [Correction(Example(**row["example"]), row["completion"], row["attempt"]) for row in rows]


def feasibility(parents, seeds, tokenizer, integrated, shared_doses):
    chat_kwargs = frozen_chat_kwargs(
        tokenizer, {"enable_thinking": False, "date_string": "11 Oct 2026"},
    )
    records = []
    for seed in seeds:
        folder = parents / str(seed)
        completion = folder / "completed.json"
        if file_digest(completion) != integrated["parent_completed_sha256"][str(seed)]:
            raise ValueError("Unexpected completion receipt")
        seal_path = folder / "commitment.json"
        done = read(completion)
        if not done["passed"] or file_digest(seal_path) != done["commitment_sha256"]:
            raise ValueError("Invalid saved commitment")
        seal = read(seal_path)
        first_rows = read(folder / "initial_pool.json")["batch"]
        if digest(first_rows) != seal["states"]["m1"]["batch_sha256"]:
            raise ValueError("First update batch differs")
        first = corrections(first_rows)
        for draw, expected in enumerate(seal["donor_sha256"]):
            path = folder / "donors" / f"{draw}.json"
            if file_digest(path) != expected:
                raise ValueError("Archived correction pool differs")
            pool = corrections(read(path)["corrections"])
            token_counts = {}
            for correction in pool:
                row = encode_completion(tokenizer, correction.example.prompt,
                                        correction.completion, 1024, chat_kwargs)
                token_counts[correction.example.id] = sum(label != -100 for label in row["labels"])
            record = matched_batches(first, pool, token_counts, seed+draw, shared_doses=shared_doses)
            if record["available"]:
                for batch in record["batches"].values():
                    batch["batch_ids"] = [c.example.id for c in batch.pop("items")]
            records.append({"seed": seed, "draw": draw, **record})
    return records


def main():
    require_compute_allocation()
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parents", "integrated", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--shared-doses", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c",
        local_files_only=True, use_fast=True,
    )
    records = feasibility(args.parents, args.seeds, tokenizer, read(args.integrated), args.shared_doses)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"passed": True, "available_draws": sum(row["available"] for row in records),
              "draws": len(records), "exploratory_training_inputs_only": True,
              "shared_doses": args.shared_doses, "records": records}
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
