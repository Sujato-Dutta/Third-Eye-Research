"""Scheduler metadata only: reserve a CPU retry and repair dependencies."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--site",choices=("vista","dgx"),required=True)
    parser.add_argument("--fixtures-sha256",required=True)
    args = parser.parse_args()
    root = Path(args.root)
    campaign = root/"runs/next_round_20261106"
    path = campaign/"ledger.json"
    ledger = json.loads(path.read_text())
    if any(r["name"] == "validate_retry" for r in ledger["reservations"]):
        raise RuntimeError("Validation retry already reserved")
    original = {r["name"]:r for r in ledger["reservations"]}
    source = (campaign/"validate.slurm").read_text()
    old = str(root/".venv/bin/python")+" "+str(campaign/"validate.py")+" "+str(campaign/"request.json")
    new = str(root/".venv/bin/python")+" "+str(campaign/"repair_package.py")+" --request "+str(campaign/"request.json")+" --fixtures-sha256 "+args.fixtures_sha256
    if source.count(old)!=1:
        raise ValueError("Unexpected validation script")
    source = source.replace(old,new).replace("te-next-validate","te-next-validate-retry")
    source = source.replace("/logs/validate-","/logs/validate-retry-")
    script = campaign/"validate_retry.slurm"
    script.write_text(source)
    sus = .33 if args.site=="vista" else 0
    if ledger["charged_and_reserved_sus"]+sus > 55:
        raise RuntimeError("First-stage budget exhausted")
    record = {"name":"validate_retry","reserved_sus":sus,"state":"SUBMITTING"}
    ledger["reservations"].append(record)
    ledger["charged_and_reserved_sus"] += sus
    path.write_text(json.dumps(ledger,indent=2)+"\n")
    outcome = subprocess.run(["sbatch","--parsable","--dependency=afterany:"+
                              original["validate"]["job_id"],str(script)],
                             text=True,capture_output=True)
    record.update({"stdout":outcome.stdout,"stderr":outcome.stderr,"exit_code":outcome.returncode})
    matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$",outcome.stdout,re.M)
    if outcome.returncode or len(matches)!=1:
        path.write_text(json.dumps(ledger,indent=2)+"\n")
        raise RuntimeError("Submission requires inspection: "+json.dumps(record))
    job = matches[0]
    record.update({"job_id":job,"state":"SUBMITTED"})
    path.write_text(json.dumps(ledger,indent=2)+"\n")
    for name in ("prepare","saved","pilots"):
        if name not in original:
            continue
        dependency = "afterok:"+job
        if name == "pilots":
            dependency += ":"+original["prepare"]["job_id"]
        subprocess.run(["scontrol","update","JobId="+original[name]["job_id"],
                        "Dependency="+dependency],check=True)
        original[name]["dependency"] = dependency
    path.write_text(json.dumps(ledger,indent=2)+"\n")
    print(json.dumps({"retry_job":job,"charged_and_reserved_sus":ledger["charged_and_reserved_sus"],
                      "dependencies_updated":True}),flush=True)


if __name__ == "__main__":
    main()
