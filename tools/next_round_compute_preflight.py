"""Allocated CPU preparation and complete validation before pilot release."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile


assert os.environ.get("SLURM_JOB_ID") and "login" not in socket.gethostname()
request = json.loads(Path(sys.argv[1]).read_text())
archive = Path(request["archive"])
source = Path(request["source"])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == request["archive_sha256"]
source.mkdir(parents=True,exist_ok=False)
with tarfile.open(archive,"r:gz") as bundle:
    for member in bundle.getmembers():
        if not member.isfile() or not (source/member.name).resolve().is_relative_to(source.resolve()):
            raise ValueError("Invalid source archive member")
    bundle.extractall(source)
sys.path[:0] = [str(source/"src"),str(source/"experiments")]
os.environ["PYTHONPATH"] = str(source/"src")+":"+str(source/"experiments")
os.environ["THIRD_EYE_MODEL_SOURCES"] = request["model_sources"]
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
import next_round_regimes  # noqa: E402
request = next_round_regimes.checked(sys.argv[1],validation=False)
print("Preparing outcome-blind inputs and checking cached models on allocated CPU",flush=True)
next_round_regimes.prepare(request)
print("Input preflight passed; running complete suite and source checks",flush=True)
subprocess.run([sys.executable,str(Path(sys.argv[1]).parent/"validate_existing.py"),sys.argv[1]],check=True)
