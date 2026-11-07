"""Scheduler-only launch using the complete input preflight and validation."""
import json
from pathlib import Path
import re
import subprocess


root = Path("/scratch/11617/sujato_ts/third_eye_research")
campaign = root/"runs/next_round_20261106"
metadata = json.loads((campaign/"clean_metadata.json").read_text())
ledger_path = campaign/"ledger.json"
ledger = json.loads(ledger_path.read_text())
assert not any(r["name"].endswith("_clean") for r in ledger["reservations"])
old = json.loads((campaign/"request.json").read_text())
request = {**old,**metadata,
           "source":str(root/("releases/next_round_"+metadata["source_tree_sha256"][:12])),
           "archive":str(campaign/"source_clean.tar.gz"),
           "output":str(campaign/"regime_clean"),
           "validation":str(campaign/"validation_clean/completed.json")}
(campaign/"regime_clean").mkdir()
(campaign/"validation_clean").mkdir()
request_path = campaign/"request_clean.json"
request_path.write_text(json.dumps(request,indent=2))
for name in ("validate","pilots"):
    text = (campaign/(name+".slurm")).read_text()
    text = text.replace(str(campaign/"request.json"),str(request_path))
    text = text.replace(str(campaign/"validate.py"),str(campaign/"compute_preflight.py"))
    text = text.replace(old["source"],request["source"])
    text = text.replace("te-next-"+name,"te-next-"+name+"-clean")
    text = text.replace("/logs/"+name+"-","/logs/"+name+"-clean-")
    (campaign/(name+"_clean.slurm")).write_text(text)
subprocess.run(["scancel","1125730"],check=True)
next(r for r in ledger["reservations"] if r["name"]=="pilots_datafix")["state"] = "CANCELLED_PENDING_FINAL_INPUT_REPAIR"
ledger_path.write_text(json.dumps(ledger,indent=2))


def submit(name,script,sus,dependency=None):
    assert ledger["charged_and_reserved_sus"]+sus <= 55
    record = {"name":name,"reserved_sus":sus,"state":"SUBMITTING"}
    ledger["reservations"].append(record)
    ledger["charged_and_reserved_sus"] = round(ledger["charged_and_reserved_sus"]+sus,8)
    ledger_path.write_text(json.dumps(ledger,indent=2))
    cmd = ["sbatch","--parsable"]
    if dependency:
        cmd += ["--dependency=afterok:"+dependency]
    cmd += [str(campaign/script)]
    outcome = subprocess.run(cmd,text=True,capture_output=True)
    record.update({"stdout":outcome.stdout,"stderr":outcome.stderr,"exit_code":outcome.returncode})
    matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$",outcome.stdout,re.M)
    if outcome.returncode or len(matches)!=1:
        ledger_path.write_text(json.dumps(ledger,indent=2))
        raise RuntimeError("Submission requires review")
    record.update({"job_id":matches[0],"state":"SUBMITTED","dependency":dependency})
    ledger_path.write_text(json.dumps(ledger,indent=2))
    print(json.dumps(record),flush=True)
    return matches[0]


validation = submit("preflight_clean","validate_clean.slurm",.33)
submit("pilots_clean","pilots_clean.slurm",12,validation)
