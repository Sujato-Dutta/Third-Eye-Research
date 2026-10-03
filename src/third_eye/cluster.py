"""Prevent research GPU execution outside a scheduler allocation."""

import os
import re
import socket


def scheduler_job_id(output):
    """Parse --parsable output even when a site wrapper prints a preamble."""
    matches = re.findall(r"^\s*(\d+)(?:;[A-Za-z0-9_.-]+)?\s*$", output, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("Scheduler output did not contain exactly one job ID")
    return matches[0]


def require_gpu_allocation(device="cuda"):
    if not str(device).startswith("cuda"):
        return
    host = socket.gethostname().lower()
    if "login" in host or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "GPU research execution requires a SLURM compute-node allocation; never run on a login node"
        )
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("SLURM job has no visible GPU allocation")
