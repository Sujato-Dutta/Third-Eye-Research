"""Download and verify the pinned public core backbone inside a CPU allocation."""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

from huggingface_hub import HfApi, hf_hub_download


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Model download and hashing require a compute allocation")
    root = Path.home() / "third_eye_dgx_20261026"
    repo = "Qwen/Qwen3-4B"
    revision = "1cfa9a7208912126459214e8b04321603b3df60c"
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Pinned backbone revision changed")
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "model.safetensors.index.json",
    }
    selected = [
        s
        for s in info.siblings
        if s.rfilename in names or s.rfilename.endswith(".safetensors")
    ]
    needed = sum(s.size for s in selected)
    used = int(
        subprocess.run(
            ["du", "-x", "-B1", "-s", str(Path.home())],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.split()[0]
    )
    if used + needed + 6_000_000_000 > 50_000_000_000:
        raise ValueError("Backbone and experiment headroom exceed 50GB policy")
    folder = root / "models/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots" / revision
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for item in selected:
        filename = hf_hub_download(
            repo, item.rfilename, revision=revision, local_dir=folder
        )
        path = Path(filename)
        with path.open("rb") as stream:
            sha = (
                hashlib.file_digest(stream, "sha256").hexdigest()
                if hasattr(hashlib, "file_digest")
                else None
            )
        if sha is None:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    hasher.update(chunk)
            sha = hasher.hexdigest()
        if item.lfs:
            if sha != item.lfs.sha256:
                raise ValueError("Official backbone LFS checksum differs")
        else:
            content = path.read_bytes()
            blob = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            if blob != item.blob_id:
                raise ValueError("Official backbone metadata Git checksum differs")
        files[item.rfilename] = {"sha256": sha, "bytes": path.stat().st_size}
        print(
            json.dumps({"verified_file": item.rfilename, "bytes": path.stat().st_size}),
            flush=True,
        )
    record = {
        "passed": True,
        "model": repo,
        "revision": revision,
        "files": files,
        "job": os.environ["SLURM_JOB_ID"],
    }
    (root / "qwen_core_download_completed.json").write_text(
        json.dumps(record, indent=2)
    )
    print(json.dumps({"pinned_core_backbone_downloaded": True, "bytes": needed}))


if __name__ == "__main__":
    main()
