"""Resolve an explicitly pinned, verified public weight source."""

import json
import os
from pathlib import Path
import re

from third_eye.io import file_digest


def resolve_source(model):
    path = os.environ.get("THIRD_EYE_MODEL_SOURCES")
    if not path:
        return model.name, model.revision, None
    mappings = json.loads(Path(path).read_text(encoding="utf-8"))
    proof_path = mappings.get(model.name)
    if not proof_path:
        return model.name, model.revision, None
    proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
    if (
        proof.get("official") != model.name
        or proof.get("verified") is not True
        or not proof.get("verified_weight_sha256")
        or not re.fullmatch(r"[0-9a-f]{40}", proof.get("original_revision", ""))
        or not re.fullmatch(r"[0-9a-f]{40}", proof.get("source_revision", ""))
        or model.revision not in {"main", proof["original_revision"]}
    ):
        raise ValueError("Model source proof does not match the requested backbone")
    source = proof.get("runtime_directory", proof["source_repository"])
    if "runtime_directory" in proof:
        root = Path(source)
        for name, expected in proof["metadata_files"].items():
            if file_digest(root / name) != expected["sha256"]:
                raise ValueError("Verified model metadata checksum mismatch")
    return (
        source,
        proof["source_revision"],
        {
            **proof,
            "proof_sha256": file_digest(proof_path),
        },
    )
