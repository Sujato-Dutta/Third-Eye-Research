"""Metadata-only submission of a dataset-loader fix, retaining prior evidence."""
import json
from pathlib import Path
import re
import subprocess

from stage import VALIDATOR


def main():
    root = Path("/scratch/11617/sujato_ts/third_eye_research")
    campaign = root/"runs/next_round_20261106"
    metadata = json.loads((campaign/"datafix_metadata.json").read_text())
    ledger_path = campaign/"ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if any(r["name"].endswith("_datafix") for r in ledger["reservations"]):
        raise RuntimeError("Data-loader-fix jobs already reserved")
    original = json.loads((campaign/"request.json").read_text())
    request = {**original, **metadata,
               "source":str(root/("releases/next_round_"+metadata["source_tree_sha256"][:12])),
               "archive":str(campaign/"source_v3.tar.gz"),
               "output":str(campaign/"regime_datafix"),
               "validation":str(campaign/"validation_datafix/completed.json")}
    (campaign/"regime_datafix").mkdir()
    (campaign/"validation_datafix").mkdir()
    new_request = campaign/"request_datafix.json"
    new_request.write_text(json.dumps(request,indent=2)+"\n")
    (campaign/"validate_datafix.py").write_text(VALIDATOR)
    for name in ("validate","prepare","pilots"):
        text = (campaign/(name+".slurm")).read_text()
        text = text.replace(str(campaign/"request.json"),str(new_request))
        text = text.replace(str(campaign/"validate.py"),str(campaign/"validate_datafix.py"))
        text = text.replace(original["source"],request["source"])
        text = text.replace("te-next-"+name,"te-next-"+name+"-datafix")
        text = text.replace("/logs/"+name+"-","/logs/"+name+"-datafix-")
        (campaign/(name+"_datafix.slurm")).write_text(text)
    old = next(r for r in ledger["reservations"] if r["name"]=="pilots")
    subprocess.run(["scancel",old["job_id"]],check=True)
    old["state"] = "CANCELLED_PENDING_DATASET_FIX"
    # Keep its full reservation until accounting confirms it never started.
    ledger_path.write_text(json.dumps(ledger,indent=2)+"\n")

    def submit(name,sus,dependency=None):
        if ledger["charged_and_reserved_sus"]+sus>55:
            raise RuntimeError("First-stage cap would be exceeded")
        record = {"name":name,"reserved_sus":sus,"state":"SUBMITTING"}
        ledger["reservations"].append(record)
        ledger["charged_and_reserved_sus"] += sus
        ledger_path.write_text(json.dumps(ledger,indent=2)+"\n")
        cmd = ["sbatch","--parsable"]
        if dependency:
            cmd += ["--dependency=afterok:"+dependency]
        cmd += [str(campaign/(name+".slurm"))]
        outcome = subprocess.run(cmd,text=True,capture_output=True)
        record.update({"stdout":outcome.stdout,"stderr":outcome.stderr,
                       "exit_code":outcome.returncode})
        matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$",outcome.stdout,re.M)
        if outcome.returncode or len(matches)!=1:
            ledger_path.write_text(json.dumps(ledger,indent=2)+"\n")
            raise RuntimeError("Submission requires inspection")
        record.update({"job_id":matches[0],"state":"SUBMITTED","dependency":dependency})
        ledger_path.write_text(json.dumps(ledger,indent=2)+"\n")
        print(json.dumps(record),flush=True)
        return matches[0]
    validation = submit("validate_datafix",.33)
    prepare = submit("prepare_datafix",.33,validation)
    submit("pilots_datafix",12,validation+":"+prepare)


if __name__ == "__main__":
    main()
