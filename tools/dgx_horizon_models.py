"""Allocated, byte-verified model staging for the additional DGX worker."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

PRIMARY = Path("/dgxa_home/se23uari167/third_eye_dgx_20261026")
ROOT = Path("/dgxa_home/se23ucse205/third_eye_horizon_20261109")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temp.replace(path)


def used():
    sizes = []
    for extra in ([], ["--apparent-size"]):
        result = subprocess.run(["du", "-x", "-B1", "-s"]+extra+[str(Path.home())],
                                capture_output=True, text=True, check=True)
        sizes.append(int(result.stdout.split()[0]))
    return max(sizes)


def main():
    if (not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower() or
            Path.home().resolve() != ROOT.parent or os.getuid() != 1339):
        raise RuntimeError("Pinned replacement account and compute allocation required")
    from huggingface_hub import hf_hub_download
    registry_path = PRIMARY/"imported/registry.json"
    if sha(registry_path) != read(PRIMARY/"prepared.json")["registry_sha256"]:
        raise ValueError("Original model registry changed")
    catalog_path = PRIMARY/"runs/scale_catalog_20261028/models.json"
    if sha(catalog_path) != read(catalog_path.parent/"completed.json")["models_sha256"]:
        raise ValueError("Pinned 8B catalogue changed")
    catalog = read(catalog_path)
    small = read(registry_path)["models"]["meta-llama/Llama-3.2-3B-Instruct"]
    plans = [{"id": "llama-3.2-3b-instruct", "official": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": small["revision"], "source_repository": small["source_repository"],
        "source_revision": small["source_revision"], "files": small["files"],
        "copy_metadata": str(PRIMARY/"models/hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"/small["revision"]),
        "original_proof": small["source_proof"]}]
    for key, official, revision, repository, source_revision in (
        ("qwen3-8b", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
         "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
        ("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659",
         "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77")):
        origin, source = catalog[official], catalog[repository]
        if origin["revision"] != revision or source["revision"] != source_revision:
            raise ValueError("Model catalogue revision mismatch")
        weights = {n: v for n, v in origin["files"].items() if n.endswith(".safetensors")}
        if not weights or any(v["sha256"] != source["files"].get(n, {}).get("sha256")
                              for n, v in weights.items()):
            raise ValueError("Public source does not match official weight bytes")
        names = set(weights) | {n for n in source["files"] if n in {
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "model.safetensors.index.json", "merges.txt", "vocab.json"}
            or "chat_template" in n}
        plans.append({"id": key, "official": official, "revision": revision,
            "source_repository": repository, "source_revision": source_revision,
            "files": {n: source["files"][n] for n in names}})
    total = sum(item["bytes"] for p in plans for item in p["files"].values())
    if used()+total+7_000_000_000 > 48_000_000_000:
        raise ValueError("Pinned weights plus seven GB output reserve do not fit storage policy")
    write(ROOT/"model_plan.json", {"models": plans, "total_model_bytes": total,
        "registry_sha256": sha(registry_path), "catalog_sha256": sha(catalog_path),
        "job_id": os.environ["SLURM_JOB_ID"]})
    mappings, proofs = {}, {}
    for plan in plans:
        folder = ROOT/"models"/plan["id"]
        folder.mkdir(parents=True, exist_ok=True)

        def install(row):
            name, item = row
            dest = folder/name
            if not dest.exists():
                if "copy_metadata" in plan and not name.endswith(".safetensors"):
                    shutil.copyfile(Path(plan["copy_metadata"])/name, dest)
                else:
                    hf_hub_download(plan["source_repository"], name,
                        revision=plan["source_revision"], local_dir=folder, token=False)
            if dest.stat().st_size != item["bytes"]:
                raise ValueError("Pinned model file size differs: "+name)
            actual = sha(dest)
            if item.get("sha256"):
                valid = actual == item["sha256"]
            else:
                blob = dest.read_bytes()
                valid = hashlib.sha1(b"blob "+str(len(blob)).encode()+b"\0"+blob).hexdigest() == item["git_blob"]
            if not valid:
                raise ValueError("Pinned model checksum differs: "+name)
            print(json.dumps({"verified": plan["id"]+"/"+name}), flush=True)
            return name, actual

        with ThreadPoolExecutor(max_workers=2) as pool:
            verified = dict(pool.map(install, sorted(plan["files"].items())))
        if "original_proof" in plan:
            proof = {k: v for k, v in plan["original_proof"].items() if k != "proof_sha256"}
            proof["runtime_directory"] = str(folder)
        else:
            proof = {"official": plan["official"], "original_revision": plan["revision"],
                "source_repository": plan["source_repository"], "source_revision": plan["source_revision"],
                "verified": True, "runtime_directory": str(folder),
                "verified_weight_sha256": {n: v for n, v in verified.items() if n.endswith(".safetensors")},
                "metadata_files": {n: {"sha256": v, "repository": plan["source_repository"],
                    "revision": plan["source_revision"]} for n, v in verified.items() if not n.endswith(".safetensors")},
                "metadata_from_pinned_public_source": True}
        path = ROOT/(plan["id"]+"_model_source.json")
        write(path, proof)
        mappings[plan["official"]] = str(path)
        proofs[plan["id"]] = {"proof_sha256": sha(path), "verified_files": verified}
    write(ROOT/"model_sources.json", mappings)
    write(ROOT/"models_completed.json", {"passed": True, "models": proofs,
        "model_sources_sha256": sha(ROOT/"model_sources.json"), "home_bytes": used(),
        "job_id": os.environ["SLURM_JOB_ID"], "driver_sha256": sha(__file__),
        "qwen4": "reuse existing verified primary cache; no duplicate weights"})


if __name__ == "__main__":
    main()
