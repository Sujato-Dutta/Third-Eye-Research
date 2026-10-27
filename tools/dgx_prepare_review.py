"""Allocated import, exact-file model download and quota checks for DGX controls."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from third_eye.cluster import require_compute_allocation
from third_eye.io import file_digest, write_json


def usage():
    return int(
        subprocess.run(
            ["du", "-x", "-B1", "-s", str(Path.home())],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.split()[0]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    require_compute_allocation()
    root = Path(args.root)
    transfer = root / "transfers"
    receipt = json.loads((transfer / "export_completed.json").read_text())
    archive = transfer / "payload.tar.gz"
    if not receipt["passed"] or file_digest(archive) != receipt["archive_sha256"]:
        raise ValueError("Original allocated export archive differs")
    imported = root / "imported"
    imported.mkdir(exist_ok=False)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                raise ValueError("Only ordinary payload files allowed")
            path = (imported / member.name).resolve()
            path.relative_to(imported.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.extractfile(member).read())
    registry_path = imported / "registry.json"
    if file_digest(registry_path) != receipt["registry_sha256"]:
        raise ValueError("Export registry differs")
    registry = json.loads(registry_path.read_text())
    for relative, item in registry["files"].items():
        path = (imported / "payload" / relative).resolve()
        path.relative_to((imported / "payload").resolve())
        if file_digest(path) != item["sha256"]:
            raise ValueError("Transferred original input changed")
    mappings, proofs = {}, {}
    from huggingface_hub import hf_hub_download

    for name, model in registry["models"].items():
        slug = name.replace("/", "--")
        folder = (
            root
            / "models/hf_cache/hub"
            / ("models--" + slug)
            / "snapshots"
            / model["revision"]
        )
        folder.mkdir(parents=True, exist_ok=True)
        required = sum(item["bytes"] for item in model["files"].values())
        if usage() + required + 6_000_000_000 > 50_000_000_000:
            raise ValueError("Model plus output allowance exceeds 50GB policy")
        for filename, item in model["files"].items():
            target = folder / filename
            if target.exists() and file_digest(target) == item["sha256"]:
                continue
            if filename.endswith(".safetensors"):
                hf_hub_download(
                    model["source_repository"],
                    filename,
                    revision=model["source_revision"],
                    local_dir=folder,
                )
            else:
                staged = transfer / "model_metadata" / slug / filename
                if staged.exists():
                    shutil.copyfile(staged, target)
                else:
                    origin = (
                        (model.get("source_proof") or {})
                        .get("metadata_sources", {})
                        .get(filename, {})
                    )
                    hf_hub_download(
                        origin.get("repository", model["source_repository"]),
                        filename,
                        revision=origin.get("revision", model["source_revision"]),
                        local_dir=folder,
                    )
            if (
                target.stat().st_size != item["bytes"]
                or file_digest(target) != item["sha256"]
            ):
                raise ValueError("Pinned original backbone file differs: " + filename)
        if model["source_proof"]:
            proof = {
                k: v for k, v in model["source_proof"].items() if k != "proof_sha256"
            }
            proof["runtime_directory"] = str(folder)
            path = root / "models" / (slug + "_source.json")
            write_json(path, proof)
            mappings[name] = str(path)
        proofs[name] = {
            "revision": model["revision"],
            "verified_files": model["files"],
            "runtime_directory": str(folder),
        }
    write_json(root / "model_sources.json", mappings)
    write_json(
        root / "sandbox.json",
        {
            "engine": "bubblewrap",
            "executable": "/usr/bin/bwrap",
            "timeout_seconds": 8,
            "memory_mb": 1024,
        },
    )
    write_json(
        root / "prepared.json",
        {
            "passed": True,
            "registry_sha256": receipt["registry_sha256"],
            "verified_models": proofs,
            "home_used_bytes": usage(),
            "limit_bytes": 50_000_000_000,
            "job": os.environ["SLURM_JOB_ID"],
        },
    )
    print(
        json.dumps(
            {
                "import_prepared": True,
                "parents": 16,
                "models": list(proofs),
                "home_used_bytes": usage(),
            }
        )
    )


if __name__ == "__main__":
    main()
