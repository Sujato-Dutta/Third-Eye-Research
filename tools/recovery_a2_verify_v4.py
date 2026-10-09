# ruff: noqa: E402
"""Load exact retained partial M1s; no retraining, harvesting or label replacement."""

import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from recovery_a2_controller_v4 import frozen, REC
from recovery_a2_runtime_v4 import RecoveryRuntime
from third_eye.config import Config
from third_eye.cluster import require_gpu_allocation
from third_eye.data.schema import Correction, Example
from third_eye.io import file_digest, write_json


def main():
    frozen()
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend
    import torch

    out = REC / "cuda_validation"
    out.mkdir(exist_ok=False)
    results = []
    for index in [20, 22, 24]:
        path = ROOT / "runs/a2/recovery_20261008_v3" / f"task_{index}.json"
        task = json.loads(path.read_text())
        assert all(file_digest(p) == h for p, h in task["input_sha256"].items())
        cfg = Config.load(task["argv"][task["argv"].index("--config") + 1])
        backend = HFBackend(cfg)
        backend.load_checkpoint(task["checkpoint"])
        parent = backend.snapshot()
        assert backend.state_hash() == task["parent_adapter_hash"]
        runtime = RecoveryRuntime(backend, task, out / f"receipts_{index}.jsonl")
        branch = next(b for b in task["branches"] if not b["complete"])
        items = json.loads(
            (
                Path(task["partial_state"]) / branch["candidate_id"] / "batch.json"
            ).read_text()
        )
        batch = [
            Correction(Example(**c["example"]), c["completion"], c["attempt"])
            for c in items
        ]

        def reject_train(*args, **kwargs):
            raise RuntimeError("Checkpoint validation attempted optimizer work")

        backend.train = reject_train
        log = runtime.train(
            batch,
            cfg.protocol.seed + task["generation"] * 1_000_000 + 300_000,
            log_path=out / f"training_{index}.jsonl",
        )
        assert backend.state_hash() == branch["t1_hash"]
        assert (
            log["optimizer_steps"] == 50
            and log["executed_optimizer_steps"] == 0
            and log["reused_partial_t1"]
        )
        backend.restore(parent)
        assert backend.state_hash() == task["parent_adapter_hash"]
        results.append(
            dict(
                index=index,
                parent_hash=task["parent_adapter_hash"],
                partial_t1_hash=branch["t1_hash"],
                checkpoint_matches=True,
                no_new_optimizer_steps=True,
                rollback_exact=True,
            )
        )
        del runtime, backend, parent
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        REC / "cuda_passed.json",
        dict(
            status="passed",
            cases=results,
            release_sha256=file_digest(REC / "release_v4.json"),
        ),
    )
    print(json.dumps(dict(status="passed", cases=results)), flush=True)


if __name__ == "__main__":
    main()
