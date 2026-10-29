"""Allocated storage inventory and pinned public 8B model catalogue."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from third_eye.cluster import require_compute_allocation

require_compute_allocation()
from huggingface_hub import HfApi

root = Path.home() / "third_eye_dgx_20261026"
output = root / "runs/scale_catalog_20261028"
output.mkdir(exist_ok=False)
api = HfApi()
repositories = ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct",
                "NousResearch/Meta-Llama-3.1-8B-Instruct",
                "unsloth/Meta-Llama-3.1-8B-Instruct"]
records = {}
for repository in repositories:
    try:
        info = api.model_info(repository, files_metadata=True, token=False)
        files = {}
        for entry in info.siblings:
            lfs = entry.lfs
            files[entry.rfilename] = {
                "bytes": entry.size,
                "git_blob": entry.blob_id,
                "sha256": lfs.sha256 if lfs else None,
            }
        records[repository] = {"revision": info.sha, "gated": info.gated,
                               "files": files}
    except Exception as exc:
        records[repository] = {"unavailable": type(exc).__name__,
                               "detail": str(exc)[:300]}
    (output / "models.json").write_text(json.dumps(records, indent=2) + "\n")
paths = [Path.home(), root, root / "models", root / "runs", root / "imported",
         root / ".venv", root / "transfers"]
usage = subprocess.run(["du", "-x", "-B1", "-s", *map(str, paths)],
                       check=True, capture_output=True, text=True).stdout
disk = subprocess.run(["df", "-B1", str(root)], check=True,
                      capture_output=True, text=True).stdout
record = {"job": os.environ["SLURM_JOB_ID"], "usage": usage, "disk": disk,
          "models_sha256": hashlib.sha256((output / "models.json").read_bytes()).hexdigest()}
(output / "completed.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
