"""User-authorized reset of replacement-account data; Slurm compute nodes only."""
import datetime
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess

EXPECTED_HOME = Path("/dgxa_home/se23ucse205")
EXPECTED_USER = "se23ucse205"
EXPECTED_UID = 1339
AUDIT = "third_eye_account_audit_20261109"
PRESERVE = {AUDIT, ".ssh", ".bashrc", ".bash_profile", ".bash_logout", ".profile",
            ".zshrc", ".zprofile", ".config", ".gitconfig", ".pam_environment"}


def write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def usage(path):
    result = subprocess.run(["du", "-x", "-B1", "-s", "--", str(path)],
                            capture_output=True, text=True, check=True, timeout=300)
    return int(result.stdout.split()[0])


def main():
    job = os.environ.get("SLURM_JOB_ID")
    if not job or "login" in socket.gethostname().lower():
        raise RuntimeError("Deletion requires an allocated compute node")
    home = Path.home().resolve(strict=True)
    if (os.getuid() != EXPECTED_UID or pwd.getpwuid(os.getuid()).pw_name != EXPECTED_USER or
            home != EXPECTED_HOME or home.stat().st_uid != EXPECTED_UID):
        raise RuntimeError("Wrong account/home; refuse any deletion")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Descriptor-based symlink-safe deletion required")
    queue = subprocess.run(["squeue", "-h", "-u", EXPECTED_USER, "-o", "%i|%T|%j"],
                           capture_output=True, text=True, check=True).stdout
    if any(row.split("|", 1)[0] != job for row in queue.splitlines() if row.strip()):
        raise RuntimeError("Other account jobs exist; refuse to delete their data")
    receipt = home/AUDIT/"reset_receipt.json"
    if receipt.exists():
        raise FileExistsError("Reset receipt exists; inspect instead of repeating deletion")
    targets = sorted(p for p in home.iterdir() if p.name not in PRESERVE)
    records = []
    # Resolve and check every recursive target before the first deletion.
    for target in targets:
        if target.parent.resolve(strict=True) != home or target.lstat().st_uid != EXPECTED_UID:
            raise RuntimeError("Deletion target escapes the named account or ownership")
        is_link = target.is_symlink()
        if not is_link:
            resolved = target.resolve(strict=True)
            if resolved == home or resolved.parent != home or os.path.ismount(resolved):
                raise RuntimeError("Unsafe absolute deletion target")
            if target.is_dir():
                for folder, dirs, files in os.walk(target, followlinks=False):
                    for name in dirs:
                        child = Path(folder)/name
                        if not child.is_symlink() and (os.path.ismount(child) or
                                child.stat().st_dev != home.stat().st_dev):
                            raise RuntimeError("Nested mount found; refuse recursive deletion")
        records.append({"path": str(target), "symlink_only": is_link,
                        "allocated_bytes": usage(target), "deleted": False})
    record = {"authorized_request": "delete everything in replacement account",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "job_id": job,
        "user": EXPECTED_USER, "home": str(home), "host": socket.gethostname(),
        "before_bytes": usage(home), "preserved": sorted(PRESERVE), "targets": records,
        "other_accounts_untouched": True, "completed": False}
    write(receipt, record)
    for target, item in zip(targets, records):
        if target.parent.resolve(strict=True) != home:
            raise RuntimeError("Target parent changed before deletion")
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            if target.resolve(strict=True).parent != home or os.path.ismount(target):
                raise RuntimeError("Target changed before recursive deletion")
            shutil.rmtree(target)
        else:
            target.unlink()
        item["deleted"] = True
        write(receipt, record)
    remaining = [p.name for p in home.iterdir() if p.name not in PRESERVE]
    if remaining:
        raise RuntimeError("Unexpected remaining data; inspect preserved receipt")
    record.update({"after_bytes": usage(home), "completed": True})
    record["reclaimed_bytes"] = record["before_bytes"]-record["after_bytes"]
    write(receipt, record)
    print(json.dumps({k: record[k] for k in ("completed", "before_bytes", "after_bytes",
        "reclaimed_bytes", "other_accounts_untouched")}), flush=True)


if __name__ == "__main__":
    main()
