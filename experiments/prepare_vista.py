"""Validate migrated data, isolated verification and pinned Vista model sources."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.data.splits import load_manifest
from third_eye.evaluation.sandbox import BubblewrapSandbox, run_bounded
from third_eye.io import file_digest, write_json
from third_eye.provenance import execution_environment, source_inventory, versions


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Vista preparation requires a compute-node allocation")
    output = Path("runs/vista_preflight") / os.environ["SLURM_JOB_ID"]
    if output.exists():
        raise FileExistsError("Preflight evidence is immutable")
    output.mkdir(parents=True)
    manifests = {}
    for task in ("math", "code"):
        path = Path(f"data/processed/v1/{task}/selection.json")
        manifest, _ = load_manifest(path)
        manifests[task] = file_digest(path)
        # Verify sealed final files by checksum without using their contents.
        final = path.with_name("final.json")
        sealed = json.loads(final.read_text())
        for split in sealed["splits"].values():
            if file_digest(final.parent / split["path"]) != split["sha256"]:
                raise RuntimeError("Migrated sealed-final file checksum mismatch")
    executable = shutil.which("bwrap")
    if not executable:
        raise RuntimeError("Vista needs an operational isolated code verifier")
    sandbox_config = {
        "engine": "bubblewrap",
        "executable": str(Path(executable).resolve()),
        "executable_sha256": file_digest(executable),
        "timeout_seconds": 8,
        "memory_mb": 1024,
    }
    config_path = Path("experiments/sandbox_vista.json")
    if config_path.exists():
        raise FileExistsError("Sandbox configuration must be new")
    write_json(config_path, sandbox_config)
    sandbox = BubblewrapSandbox(executable, executable_sha256=file_digest(executable))
    probe = run_bounded(
        sandbox.command(
            None,
            [
                "-c",
                "from pathlib import Path; import socket; assert not any(Path(p).exists() for p in ('/home1','/work','/scratch')); s=socket.socket(); s.settimeout(1); assert s.connect_ex(('1.1.1.1',53)) != 0; print('ISOLATED')",
            ],
        ),
        timeout=8,
        kill_group=True,
    )
    if probe.returncode or probe.stdout.strip() != b"ISOLATED":
        raise RuntimeError("Vista sandbox filesystem/network isolation failed")
    _, code = load_manifest("data/processed/v1/code/selection.json")
    example = code["train"][0]
    if sandbox.verify(example, "import sys; sys.exit(0)") or sandbox.verify(
        example, "while True: pass"
    ):
        raise RuntimeError("Vista sandbox accepted early exit or unbounded execution")
    os.environ["THIRD_EYE_SANDBOX_CONFIG"] = str(config_path.resolve())
    subprocess.run([sys.executable, "experiments/profile_verification.py"], check=True)
    from huggingface_hub import snapshot_download

    qwen_revision = "1cfa9a7208912126459214e8b04321603b3df60c"
    snapshot_download(
        "Qwen/Qwen3-4B",
        revision=qwen_revision,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.jinja",
            "tokenizer*",
            "merges.txt",
            "vocab.*",
        ],
    )
    proof = Path("models/mirror_verification/llama3_2_3b_vista.json")
    subprocess.run(
        [
            sys.executable,
            "experiments/prefetch_mirror.py",
            "--official",
            "meta-llama/Llama-3.2-3B-Instruct",
            "--official-revision",
            "0cb88a4f764b7a12671c53f0838cd831a0843b95",
            "--mirror",
            "unsloth/Llama-3.2-3B-Instruct",
            "--mirror-revision",
            "006f5dcd1393c3add266de40994ba96225e9689d",
            "--metadata-source",
            "NousResearch/Meta-Llama-3.1-8B-Instruct",
            "--output",
            str(proof),
        ],
        check=True,
    )
    write_json(
        "experiments/model_sources_vista.json",
        {"meta-llama/Llama-3.2-3B-Instruct": str(proof.resolve())},
    )
    write_json(
        output / "passed.json",
        {
            "status": "passed",
            "selection_manifest_sha256": manifests,
            "sandbox": sandbox_config,
            "execution": execution_environment(),
            "dependencies": versions(),
            **source_inventory(),
            "qwen_revision": qwen_revision,
            "llama_source_proof_sha256": file_digest(proof),
        },
    )


if __name__ == "__main__":
    main()
