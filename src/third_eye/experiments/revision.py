"""Fixed paired controls with strict, explicitly portable adapter restoration."""

import json
from pathlib import Path

from third_eye.data.schema import Correction
from third_eye.io import digest, file_digest


def semantic_source(proof):
    if proof is None:
        return None
    # Only physical placement and the checksum changed by physical placement
    # may differ. Official revisions, weight hashes and metadata stay identical.
    return {
        k: v for k, v in proof.items() if k not in {"runtime_directory", "proof_sha256"}
    }


def check_portable_identity(saved, backend):
    if (
        saved["config_hash"] != backend.config.fingerprint
        or saved["resolved_revision"] != backend.resolved_revision
        or saved["input_rendering"] != backend.input_rendering
        or semantic_source(saved.get("model_source"))
        != semantic_source(backend.source_proof)
    ):
        raise ValueError(
            "Transferred checkpoint backbone/configuration/rendering differs"
        )


def restore_portable(backend, directory):
    from peft.utils.save_and_load import load_peft_weights

    directory = Path(directory)
    saved = json.loads((directory / "state.json").read_text())
    check_portable_identity(saved, backend)
    if not saved["weights"]:
        raise ValueError("Transferred adapter has no weights")
    for name, sha in saved["weights"].items():
        path = directory / name
        path.resolve().relative_to(directory.resolve())
        if file_digest(path) != sha:
            raise ValueError("Transferred adapter checksum differs")
    backend.restore(load_peft_weights(directory, device="cpu"))
    if backend.state_hash() != saved["adapter_hash"]:
        raise ValueError("Transferred adapter state differs after restoration")
    return saved


def pooled_corrections(batches):
    chosen = {}
    for batch in batches:
        for item in batch:
            previous = chosen.get(item.example.id)
            if previous is not None and previous.example != item.example:
                raise ValueError("One problem ID identifies different examples")
            if previous is None or digest(item.to_dict()) < digest(previous.to_dict()):
                chosen[item.example.id] = item
    return tuple(chosen[key] for key in sorted(chosen))


def reference_corrections(batch):
    return tuple(Correction(item.example, item.example.answer, 1) for item in batch)
