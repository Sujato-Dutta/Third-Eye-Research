"""Download a public mirror and verify weights against original Hub hashes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.io import file_digest, write_json
from third_eye.storage import check_storage


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official", required=True)
    p.add_argument("--mirror", required=True)
    p.add_argument("--official-revision")
    p.add_argument("--mirror-revision")
    p.add_argument("--output", required=True)
    p.add_argument("--metadata-source", action="append", default=[])
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        p.error("Downloads require a compute-node SLURM allocation")
    if Path(a.output).exists():
        p.error("Source proofs are immutable; choose a new output path")
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download

    api = HfApi()
    original = api.model_info(
        a.official, revision=a.official_revision, files_metadata=True
    )
    mirror = api.model_info(a.mirror, revision=a.mirror_revision, files_metadata=True)
    if not all(re.fullmatch(r"[0-9a-f]{40}", info.sha) for info in (original, mirror)):
        raise ValueError("Sources require immutable model revisions")
    expected = {
        item.rfilename: item.lfs.sha256
        for item in original.siblings
        if item.rfilename.endswith(".safetensors") and item.lfs
    }
    offered = {
        item.rfilename: item.lfs.sha256
        for item in mirror.siblings
        if item.rfilename.endswith(".safetensors") and item.lfs
    }
    if not expected or expected != offered:
        raise ValueError("Public source weight hashes differ from the original model")
    weight_bytes = sum(
        item.lfs.size for item in original.siblings if item.rfilename in expected
    )
    check_storage(Path.cwd(), 50, headroom_gb=(weight_bytes + 1024**3) / 1e9)
    snapshot = Path(
        snapshot_download(
            a.mirror,
            revision=mirror.sha,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.jinja",
                "*.tiktoken",
                "merges.txt",
                "vocab.*",
                "tokenizer*",
            ],
        )
    )
    verified = {name: file_digest(snapshot / name) for name in expected}
    if verified != expected:
        raise ValueError(
            "Downloaded weight contents fail original SHA-256 verification"
        )
    runtime = Path("models/verified_sources") / (
        a.official.replace("/", "--") + "--" + original.sha
    )
    if runtime.exists():
        raise FileExistsError("Verified runtime directories are immutable")
    runtime.mkdir(parents=True)
    donors = [api.model_info(repo, files_metadata=True) for repo in a.metadata_source]
    donor_sources = {}
    for item in original.siblings:
        if "/" in item.rfilename:
            continue
        local = snapshot / item.rfilename
        if local.is_file():
            (runtime / item.rfilename).symlink_to(local.resolve())
        if not item.rfilename.endswith((".json", ".model", ".jinja", ".tiktoken")):
            continue
        for donor in donors:
            offered_file = next(
                (s for s in donor.siblings if s.rfilename == item.rfilename), None
            )
            same = offered_file and (
                offered_file.lfs and offered_file.lfs.sha256 == item.lfs.sha256
                if item.lfs
                else offered_file.blob_id == item.blob_id
            )
            if same:
                downloaded = Path(
                    hf_hub_download(donor.id, item.rfilename, revision=donor.sha)
                )
                verified_metadata = (
                    file_digest(downloaded) == item.lfs.sha256
                    if item.lfs
                    else git_blob(downloaded) == item.blob_id
                )
                if not verified_metadata:
                    raise ValueError(
                        "Metadata source contents fail original Git blob verification"
                    )
                target = runtime / item.rfilename
                if target.exists():
                    target.unlink()
                shutil.copyfile(downloaded, target)
                donor_sources[item.rfilename] = {
                    "repository": donor.id,
                    "revision": donor.sha,
                }
                break
    snapshot = runtime
    metadata = {}
    for item in original.siblings:
        local = snapshot / item.rfilename
        if item.rfilename in expected or not local.is_file():
            continue
        if not item.rfilename.endswith((".json", ".model", ".jinja", ".tiktoken")):
            continue
        metadata[item.rfilename] = {
            "sha256": file_digest(local),
            "byte_identical_to_original": (
                file_digest(local) == item.lfs.sha256
                if item.lfs
                else git_blob(local) == item.blob_id
            ),
        }
    # The vocabulary must be unchanged; chat/config differences remain explicit.
    vocab = metadata.get("tokenizer.json") or metadata.get("tokenizer.model")
    if not vocab or not vocab["byte_identical_to_original"]:
        raise ValueError("Public source tokenizer vocabulary differs from the original")
    proof = {
        "official": a.official,
        "original_revision": original.sha,
        "source_repository": a.mirror,
        "source_revision": mirror.sha,
        "runtime_directory": str(runtime.resolve()),
        "metadata_sources": donor_sources,
        "verified": True,
        "verified_weight_sha256": verified,
        "metadata_files": metadata,
        "configuration": json.loads((snapshot / "config.json").read_text()),
        "configuration_differences": [
            name
            for name, item in metadata.items()
            if not item["byte_identical_to_original"]
        ],
        "host": socket.gethostname(),
        "job_id": os.environ["SLURM_JOB_ID"],
    }
    write_json(a.output, proof)
    print(
        json.dumps(
            {
                "official": a.official,
                "source": a.mirror,
                "verified": True,
                "configuration_differences": proof["configuration_differences"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
