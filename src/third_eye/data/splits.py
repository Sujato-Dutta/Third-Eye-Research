"""Split files are hashed once; no benchmark downloading or test leakage."""

from dataclasses import asdict
import json
from pathlib import Path

from third_eye.data.schema import Example, ROLES
from third_eye.io import digest, file_digest, write_json


def read_examples(path, role):
    examples = [
        Example(**json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not examples or any(ex.split != role for ex in examples):
        raise ValueError(f"Empty or incorrectly tagged {role} split")
    if len({ex.id for ex in examples}) != len(examples):
        raise ValueError(f"Duplicate IDs in {role}")
    if len({ex.prompt_hash for ex in examples}) != len(examples):
        raise ValueError(f"Duplicate prompts in {role}")
    return examples


def validate_splits(splits):
    if set(splits) != set(ROLES):
        raise ValueError("Require train plus three distinct development/proxy splits")
    seen_ids, seen_prompts = set(), set()
    for role in ROLES:
        for ex in splits[role]:
            if ex.id in seen_ids or ex.prompt_hash in seen_prompts:
                raise ValueError("Cross-split ID or normalized-prompt leakage")
            seen_ids.add(ex.id)
            seen_prompts.add(ex.prompt_hash)
    tasks = {ex.task for ex in splits["train"]}
    if len(tasks) != 1:
        raise ValueError("Run one training task family at a time")


def create_manifest(paths, output):
    if Path(output).exists():
        raise FileExistsError("Manifests are immutable; use a new path/version")
    splits = {role: read_examples(paths[role], role) for role in ROLES}
    validate_splits(splits)
    entries = {
        role: {
            "path": str(Path(paths[role]).resolve()),
            "sha256": file_digest(paths[role]),
            "count": len(splits[role]),
            "examples_hash": digest([asdict(ex) for ex in splits[role]]),
        }
        for role in ROLES
    }
    manifest = {"schema_version": 1, "splits": entries}
    write_json(output, manifest)
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("splits", {})) != set(
        ROLES
    ):
        raise ValueError("Unsupported split manifest")
    splits = {}
    for role, entry in manifest["splits"].items():
        if file_digest(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Immutable split changed: {role}")
        splits[role] = read_examples(entry["path"], role)
        if (
            len(splits[role]) != entry["count"]
            or digest([asdict(ex) for ex in splits[role]]) != entry["examples_hash"]
        ):
            raise ValueError(f"Manifest contents disagree: {role}")
    validate_splits(splits)
    return manifest, splits
