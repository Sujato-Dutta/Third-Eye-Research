"""Resume the explicitly authorized account reset across read-only cache dirs."""
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import stat
import subprocess

from reset import AUDIT, EXPECTED_HOME, EXPECTED_UID, EXPECTED_USER, PRESERVE, usage, write


def writable_directory(path, target, device):
    if path.is_symlink():
        return
    resolved = path.resolve(strict=True)
    resolved.relative_to(target.resolve(strict=True))
    details = path.stat()
    if (details.st_uid != EXPECTED_UID or details.st_dev != device or os.path.ismount(path)):
        raise RuntimeError("Refuse permission changes outside owned, unmounted deletion targets")
    os.chmod(path, stat.S_IMODE(details.st_mode) | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def main():
    job = os.environ.get("SLURM_JOB_ID")
    home = Path.home().resolve(strict=True)
    if (not job or "login" in socket.gethostname().lower() or home != EXPECTED_HOME or
            os.getuid() != EXPECTED_UID or pwd.getpwuid(os.getuid()).pw_name != EXPECTED_USER):
        raise RuntimeError("Wrong account, home or node; refuse cleanup")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Symlink-safe deletion required")
    queue = subprocess.run(["squeue", "-h", "-u", EXPECTED_USER, "-o", "%i"],
                           capture_output=True, text=True, check=True).stdout
    if any(row.strip() != job for row in queue.splitlines() if row.strip()):
        raise RuntimeError("Other jobs exist; refuse cleanup")
    receipt = home/AUDIT/"reset_receipt.json"
    record = json.loads(receipt.read_text())
    if (record["completed"] or record["home"] != str(EXPECTED_HOME) or
            record["user"] != EXPECTED_USER or record["job_id"] != "35059"):
        raise ValueError("Expected original incomplete reset receipt required")
    targets = [Path(item["path"]) for item in record["targets"]]
    for target, item in zip(targets, record["targets"]):
        if target.parent.resolve(strict=True) != home or target.name in PRESERVE:
            raise ValueError("Retained cleanup target escaped the explicit account scope")
        exists = target.exists() or target.is_symlink()
        if item["deleted"] and exists:
            raise ValueError("Previously deleted data reappeared; do not erase new work")
        if exists:
            if target.lstat().st_uid != EXPECTED_UID:
                raise ValueError("Unexpected target ownership")
            if not target.is_symlink() and (target.resolve(strict=True).parent != home or os.path.ismount(target)):
                raise ValueError("Unsafe resolved deletion target")
    saved = receipt.with_name("reset_receipt_before_finish_"+job+".json")
    if saved.exists():
        raise FileExistsError("Preserve previous cleanup retry")
    shutil.copyfile(receipt, saved)
    record["finish_job_id"] = job
    record["permission_repair"] = "add owner rwx only to owned directories inside approved deletion targets"
    write(receipt, record)
    device = home.stat().st_dev
    for target, item in zip(targets, record["targets"]):
        if item["deleted"]:
            continue
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            writable_directory(target, target, device)
            for folder, dirs, files in os.walk(target, followlinks=False):
                writable_directory(Path(folder), target, device)
                for name in dirs:
                    writable_directory(Path(folder)/name, target, device)
            if target.resolve(strict=True).parent != home or os.path.ismount(target):
                raise ValueError("Target changed before recursive deletion")
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        item["deleted"] = True
        write(receipt, record)
    if any(p.name not in PRESERVE for p in home.iterdir()):
        raise ValueError("Unexpected remaining data; inspect receipt")
    record.update({"completed": True, "after_bytes": usage(home)})
    record["reclaimed_bytes"] = record["before_bytes"]-record["after_bytes"]
    write(receipt, record)
    print(json.dumps({k: record[k] for k in ("completed", "before_bytes", "after_bytes",
                                           "reclaimed_bytes", "other_accounts_untouched")}), flush=True)


if __name__ == "__main__":
    main()
