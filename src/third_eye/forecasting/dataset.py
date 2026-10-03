"""Whole-trajectory partitions, merging trajectories that share states."""

from collections import defaultdict
import json
import math
from pathlib import Path
import random

from third_eye.io import digest


def read_records(paths):
    records, seen = [], {}
    for path in paths:
        path = Path(path)
        # Older resumed slices inherit their original run's trajectory.
        directory = path.parent
        visited = set()
        while (directory / "run.json").exists():
            if directory in visited:
                raise ValueError("Cyclic trajectory resume provenance")
            visited.add(directory)
            provenance = json.loads(
                (directory / "run.json").read_text(encoding="utf-8")
            )
            if provenance.get("trajectory_id") or not provenance.get("resume"):
                break
            checkpoint = Path(provenance["resume"])
            if not checkpoint.is_absolute():
                checkpoint = directory.parent / checkpoint
            directory = checkpoint.parent.parent
        default_trajectory = str(directory.resolve())
        if (directory / "run.json").exists():
            default_trajectory = provenance.get("trajectory_id", default_trajectory)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["state_id"], record["candidate_id"])
            if key in seen:
                if digest(seen[key]) != digest(record):
                    raise ValueError("Conflicting duplicate candidate record")
                continue
            seen[key] = record
            record = dict(
                record, trajectory_id=record.get("trajectory_id", default_trajectory)
            )
            records.append(record)
    states = defaultdict(list)
    if len({r.get("protocol_amendment", "original") for r in records}) > 1:
        raise ValueError("Different protocol amendments cannot enter a pooled dataset")
    for record in records:
        states[record["state_id"]].append(record)
    for state, rows in states.items():
        if len(rows) != 3 or len({r["candidate_id"] for r in rows}) != 3:
            raise ValueError(f"Incomplete K=3 state: {state}")
        for field in (
            "model",
            "seed",
            "config_hash",
            "manifest_hash",
            "parent_adapter_hash",
            "generation",
        ):
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f"Unmatched {field} in state {state}")
        if len({r["candidate_size"] for r in rows}) != 1:
            raise ValueError("Candidate sizes differ")
    return records


def split_records(records, seed=42, held_out_models=()):
    """Union all shared states; train/validation/test never share a trajectory.

    Held-out families/scales go only to test. At least three independent
    components are needed; tiny pilot slices are rejected rather than leaked.
    """
    parent = {r["trajectory_id"]: r["trajectory_id"] for r in records}

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    states = {}
    for r in records:
        t, s = r["trajectory_id"], r["state_id"]
        if s in states:
            parent[root(t)] = root(states[s])
        states[s] = t
    groups = defaultdict(list)
    for r in records:
        groups[root(r["trajectory_id"])].append(r)
    held, core = [], []
    for group in groups.values():
        if any(r["model"] in held_out_models for r in group):
            held.extend(group)
        else:
            core.append(group)
    if len(core) < 3:
        raise ValueError(
            "Need at least three independent core trajectories for train/validation/test"
        )
    random.Random(seed).shuffle(core)
    nval = max(1, math.ceil(len(core) * 0.20))
    ntest = max(1, math.ceil(len(core) * 0.20))
    ntrain = len(core) - nval - ntest
    parts = {
        "train": sum(core[:ntrain], []),
        "validation": sum(core[ntrain : ntrain + nval], []),
        "test": sum(core[ntrain + nval :], []) + held,
    }
    return parts
