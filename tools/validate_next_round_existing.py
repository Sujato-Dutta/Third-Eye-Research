"""Validate an already unpacked release, checking every archived source file."""
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
source = Path(request["source"])
archive = Path(request["archive"])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == request["archive_sha256"]
with tarfile.open(archive,"r:gz") as bundle:
    for member in bundle.getmembers():
        target = source/member.name
        if not member.isfile() or not target.resolve().is_relative_to(source.resolve()):
            raise ValueError("Invalid source archive member")
        if hashlib.sha256(bundle.extractfile(member).read()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise ValueError("Extracted source or fixture differs from archive")
sys.path[:0] = [str(source/"src"),str(source/"experiments")]
from third_eye.provenance import source_inventory  # noqa: E402
assert source_inventory(source)["source_tree_sha256"] == request["source_tree_sha256"]
base = Path(request["output"])/"validation"
base.mkdir(parents=True,exist_ok=True)
receipt = Path(request["validation"])
if receipt.exists():
    old = json.loads(receipt.read_text())
    receipt.with_name("completed_failed_"+old["job_id"]+".json").write_text(receipt.read_text())
result = {"passed":False,"source_tree_sha256":request["source_tree_sha256"],
          "archive_sha256":request["archive_sha256"],"job_id":os.environ["SLURM_JOB_ID"],
          "host":socket.gethostname(),"all_archived_files_verified":True}
try:
    subprocess.run([sys.executable,"-m","pytest","-q","--basetemp",
                    str(base/("fixtures_"+os.environ["SLURM_JOB_ID"]))],cwd=source,check=True)
    subprocess.run([sys.executable,"-m","ruff","check","src","experiments","tests"],cwd=source,check=True)
    result["passed"] = True
finally:
    receipt.write_text(json.dumps(result,indent=2)+"\n")
