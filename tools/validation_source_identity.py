"""Describe direct pinned sources and verified mirrors without inventing proofs."""


def model_source_identity(config, backend):
    revision = backend.resolved_revision
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
        or revision != config.model.revision
    ):
        raise ValueError("Validation requires the declared pinned model revision")
    proof = backend.source_proof
    if proof is None:
        return {
            "model": config.model.name,
            "resolved_revision": revision,
            "load_route": "pinned_hugging_face",
            "mirror_proof_sha256": None,
        }
    if (
        not proof.get("verified")
        or proof.get("official") != config.model.name
        or proof.get("original_revision") != revision
        or not proof.get("proof_sha256")
    ):
        raise ValueError("Invalid verified-mirror source identity")
    return {
        "model": config.model.name,
        "resolved_revision": revision,
        "load_route": "verified_mirror",
        "mirror_proof_sha256": proof["proof_sha256"],
    }
