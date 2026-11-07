"""CPU validation retry for omitted static JSON test fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request",required=True)
    parser.add_argument("--fixtures-sha256",required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID") and "login" not in socket.gethostname()
    request = json.loads(Path(args.request).read_text())
    source = Path(request["source"])
    archive = Path(request["output"])/"static_config_fixtures.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != args.fixtures_sha256:
        raise ValueError("Supplemental test-fixture archive changed")
    with tarfile.open(archive,"r:gz") as bundle:
        for member in bundle.getmembers():
            target = source/member.name
            if (not member.isfile() or not member.name.startswith("experiments/configs/") or
                    not target.resolve().is_relative_to((source/"experiments/configs").resolve()) or
                    target.suffix != ".json"):
                raise ValueError("Only static configuration fixtures permitted")
            if target.exists():
                raise FileExistsError("Repair must not replace existing source inputs")
        bundle.extractall(source)
    sys.path[:0] = [str(source/"src"),str(source/"experiments")]
    from third_eye.provenance import source_inventory
    assert source_inventory(source)["source_tree_sha256"] == request["source_tree_sha256"]
    receipt = Path(request["validation"])
    if receipt.exists():
        shutil.copyfile(receipt, receipt.with_name("completed_failed_original.json"))
    result = {"passed":False,"source_tree_sha256":request["source_tree_sha256"],
              "archive_sha256":request["archive_sha256"],
              "static_fixture_archive_sha256":args.fixtures_sha256,
              "job_id":os.environ["SLURM_JOB_ID"],"host":socket.gethostname(),
              "repair":"Include omitted JSON test fixtures; scientific source unchanged"}
    try:
        subprocess.run([sys.executable,"-m","pytest","-q","--basetemp",
                        str(receipt.parent/("fixtures_retry_"+os.environ["SLURM_JOB_ID"]))],
                       cwd=source,check=True)
        subprocess.run([sys.executable,"-m","ruff","check","src","experiments","tests"],
                       cwd=source,check=True)
        result["passed"] = True
    finally:
        receipt.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
