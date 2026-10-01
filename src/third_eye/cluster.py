"""Prevent research GPU execution outside a scheduler allocation."""

import os
import socket


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
