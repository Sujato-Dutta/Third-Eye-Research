"""Inspect completed seed-1042 pilots; scientific acceptance requires review."""

import json
import math
from pathlib import Path

from third_eye.config import Config
from third_eye.forecasting.dataset import read_records
from third_eye.io import digest, file_digest
from third_eye.updates.candidates import batch_hash
from third_eye.data.schema import Correction, Example

HEADS = ("target", "ood", "retention")
STREAMS = {
    "qwen3_4b:math",
    "qwen3_4b:code",
    "llama3_2_3b:math",
    "llama3_2_3b:code",
}


def inspect_pilots(tasks):
    """Return immutable evidence and errors, never a phenomenon-gate decision."""
    evidence, streams, errors = {}, [], []

    def load(path):
        path = Path(path)
        evidence[str(path)] = file_digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def corrections(items):
        return [
            Correction(Example(**item["example"]), item["completion"], item["attempt"])
            for item in items
        ]

    def pool(path, protocol):
        value = load(path)
        items, stats = value["items"], value["stats"]
        ids = [item["example"]["id"] for item in items]
        require(len(ids) == len(set(ids)), "Correction pool contains duplicate IDs")
        require(stats["harvest_complete"] is True, "Incomplete correction harvest")
        require(
            stats["correction_attempt_limit"] == protocol.correction_attempts,
            "Harvest retry budget changed",
        )
        require(
            stats["verified_corrections"] == len(items),
            "Correction pool count disagrees",
        )
        require(
            all(1 <= item["attempt"] <= protocol.correction_attempts for item in items),
            "Invalid correction attempt",
        )
        return {item["example"]["id"]: item for item in items}, stats

    require(
        len(tasks) == 4 and {t["stream"] for t in tasks} == STREAMS,
        "Require exactly four A1 streams",
    )
    for task in tasks:
        output = Path(task["output"])
        stream = {"stream": task["stream"], "output": str(output), "status": "invalid"}
        try:
            command = task["commands"][0]
            config_path = Path(command[command.index("--config") + 1])
            evidence[str(config_path)] = file_digest(config_path)
            cfg = Config.load(config_path)
            p = cfg.protocol
            require(task["seed"] == p.seed == 1042, "Initial pilots must use seed 1042")
            require(
                p.amendment == "A1" and p.candidates == 3 and p.candidate_size == 2,
                "Invalid A1 candidate protocol",
            )
            completed, run = load(output / "completed.json"), load(output / "run.json")
            require(
                completed["status"] == "complete"
                and completed["completed_states"]
                == completed["requested_generations"]
                == 1,
                "Pilot did not complete one state",
            )
            require(
                run["config_hash"] == cfg.fingerprint
                and digest(run["config"]) == cfg.fingerprint,
                "Pilot configuration changed",
            )
            labels_path = output / "meta_labels.jsonl"
            evidence[str(labels_path)] = file_digest(labels_path)
            rows = read_records([labels_path])
            require(
                len(rows) == 3 and len({r["state_id"] for r in rows}) == 1,
                "Incomplete one-state K=3 labels",
            )
            state = output / "states" / rows[0]["state_id"]
            root_pool, root_stats = pool(state / "correction_pool.json", p)
            compositions, terminals = set(), 0
            for row in rows:
                require(
                    row["protocol_amendment"] == "A1" and row["seed"] == 1042,
                    "Labels use a different protocol",
                )
                require(
                    row["config_hash"] == cfg.fingerprint
                    and row["manifest_hash"] == run["manifest_hash"]
                    and row["model"] == cfg.model.name,
                    "Labels disagree with run provenance",
                )
                directory = state / row["candidate_id"]
                batch = load(directory / "batch.json")
                ids = [item["example"]["id"] for item in batch]
                require(
                    len(ids) == len(set(ids)) == p.candidate_size,
                    "Invalid within-batch composition",
                )
                require(
                    all(item == root_pool.get(item["example"]["id"]) for item in batch),
                    "Candidate uses an item outside the verified pool",
                )
                require(
                    batch_hash(corrections(batch)) == row["batch_hash"],
                    "Candidate batch hash disagrees",
                )
                compositions.add(tuple(sorted(ids)))
                continuation_pool, stats = pool(directory / "continuation_pool.json", p)
                require(
                    stats == row["continuation"]["pool_statistics"]
                    and stats["training_prompts"] == root_stats["training_prompts"],
                    "Continuation harvest disagrees with labels",
                )
                available = row["continuation_available"]
                require(
                    type(available) is int and available in (0, 1),
                    "Missing continuation availability",
                )
                require(
                    available
                    == row["continuation"]["available"]
                    == int(len(continuation_pool) >= p.candidate_size),
                    "Continuation availability disagrees with pool",
                )
                require(
                    row["runtime"]["candidate"]["optimizer_steps"]
                    == cfg.training.max_steps,
                    "Candidate training budget changed",
                )
                continuation_batch = load(directory / "continuation_batch.json")
                if available:
                    continuation_ids = [
                        item["example"]["id"] for item in continuation_batch
                    ]
                    require(
                        len(continuation_ids)
                        == len(set(continuation_ids))
                        == p.candidate_size,
                        "Invalid continuation batch",
                    )
                    require(
                        all(
                            item == continuation_pool.get(item["example"]["id"])
                            for item in continuation_batch
                        ),
                        "Continuation uses an unverified item",
                    )
                    require(
                        batch_hash(corrections(continuation_batch))
                        == row["continuation"]["batch_hash"],
                        "Continuation batch hash disagrees",
                    )
                    require(
                        row["runtime"]["continuation"]["optimizer_steps"]
                        == cfg.training.max_steps,
                        "Continuation training budget changed",
                    )
                else:
                    terminals += 1
                    require(
                        not continuation_batch
                        and row["continuation"]["terminal_reason"]
                        == "correction_scarcity",
                        "Invalid terminal metadata",
                    )
                    require(
                        row["runtime"]["continuation"]["optimizer_steps"] == 0,
                        "Terminal branch trained",
                    )
                    require(
                        row["evaluation"]["t2"] == row["evaluation"]["t1"]
                        and row["labels"]["h2"] == row["labels"]["h1"],
                        "Terminal H=2 is not identity",
                    )
                for horizon, point in ((1, "t1"), (2, "t2")):
                    for head in HEADS:
                        parent, value = (
                            row["evaluation"]["parent"][head],
                            row["evaluation"][point][head],
                        )
                        require(
                            all(
                                type(v) in (float, int)
                                and math.isfinite(v)
                                and 0 <= v <= 1
                                for v in (parent, value)
                            ),
                            "Missing or invalid evaluation metrics",
                        )
                        require(
                            math.isclose(
                                row["labels"][f"h{horizon}"][head],
                                value - parent,
                                abs_tol=1e-12,
                            ),
                            "Horizon delta disagrees with evaluation",
                        )
                    utility = sum(
                        w * row["labels"][f"h{horizon}"][h]
                        for w, h in zip(p.utility_weights, HEADS)
                    )
                    require(
                        math.isclose(
                            row["labels"][f"utility_h{horizon}"], utility, abs_tol=1e-12
                        ),
                        "Utility disagrees with consequences",
                    )
            require(
                len(compositions) == 3,
                "Candidates do not have three distinct compositions",
            )
            stream.update(
                status="valid",
                complete_states=1,
                candidate_branches=3,
                terminal_continuations=terminals,
                terminal_fraction=terminals / 3,
                metrics_populated=True,
                distinct_compositions=3,
                evaluations=[r["evaluation"] for r in rows],
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            errors.append({"stream": task["stream"], "error": str(error)})
        streams.append(stream)
    terminals = sum(s.get("terminal_continuations", 0) for s in streams)
    branches = sum(s.get("candidate_branches", 0) for s in streams)
    return {
        "status": "valid" if not errors else "inspection_failed",
        "streams": streams,
        "errors": errors,
        "terminal_continuations": terminals,
        "candidate_branches": branches,
        "terminal_fraction": terminals / branches if branches else None,
        "all_terminal_streams": [
            s["stream"] for s in streams if s.get("terminal_fraction") == 1
        ],
        "evidence_hashes": evidence,
        "scientific_review_required": True,
        "note": "Inspect terminal dominance without changing A1 or applying an invented cutoff; this report is not Gate 1.",
    }


def review_allows_calibration(report, report_path, reviewed_hash):
    return report["status"] == "valid" and reviewed_hash == file_digest(report_path)
