"""Reuse completed, hash-verified branches of a timed-out A2 state."""

from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time
from third_eye.io import file_digest

from third_eye.data.schema import Correction, Example
from third_eye.evaluation.runner import Consequences
from third_eye.io import append_jsonl
from third_eye.updates.candidates import batch_hash


class RecoveryRuntime:
    def __init__(self, backend, context, receipts):
        self.backend = backend
        self.context = context
        self.receipts = Path(receipts)
        self.pools, self.evaluations, self.features, self.trains = {}, {}, {}, {}
        self.replay_checks = {}
        self.cfg = backend.config
        self.parent = context["parent_adapter_hash"]
        self.seed = self.cfg.protocol.seed + context["generation"] * 1_000_000
        root = Path(context["partial_state"])
        pool = json.loads((root / "correction_pool.json").read_text())
        self.pools[(self.parent, self.seed)] = (pool, root / "correction_pool.json")
        for item in context["branches"]:
            branch = root / item["candidate_id"]
            key = (self.parent, item["batch_hash"], self.seed + 300_000, 50)
            if not item["complete"]:
                if not item.get("t1_hash"):
                    raise RuntimeError(
                        "V4 requires the saved, fully trained partial-branch M1"
                    )
                saved = json.loads((branch / "t1/state.json").read_text())
                original_log = branch / "training_t1.jsonl"
                rows = [
                    json.loads(line) for line in original_log.read_text().splitlines()
                ]
                if (
                    saved["config_hash"] != self.cfg.fingerprint
                    or saved["adapter_hash"] != item["t1_hash"]
                ):
                    raise RuntimeError("Partial M1 protocol/hash changed")
                if [r["step"] for r in rows] != list(range(1, 51)) or any(
                    not math.isfinite(r["loss"])
                    or not math.isfinite(r["gradient_norm"])
                    for r in rows
                ):
                    raise RuntimeError(
                        "Partial M1 lacks fifty completed finite training steps"
                    )
                if any(
                    file_digest(branch / "t1" / name) != sha
                    for name, sha in saved["weights"].items()
                ):
                    raise RuntimeError("Partial M1 weight file changed")
                self.trains[key] = (
                    branch / "t1",
                    item["t1_hash"],
                    {
                        "optimizer_steps": 50,
                        "examples": 2,
                        "seed": self.seed + 300_000,
                        "loss_first": rows[0]["loss"],
                        "loss_last": rows[-1]["loss"],
                        "executed_optimizer_steps": 0,
                        "reused_partial_t1": True,
                        "source_training_log_sha256": file_digest(original_log),
                        "original_training_seconds": None,
                        "seconds_semantics": "measured checkpoint-load time; original producer timer unavailable",
                    },
                    original_log,
                )
                continue
            row = json.loads((branch / "record.json").read_text())
            self.features[(self.parent, row["batch_hash"], self.seed + 300_000)] = row[
                "precommit"
            ]["features"]
            for when, state in [
                ("parent", self.parent),
                ("t1", row["candidate_adapter_hash"]),
                ("t2", row["continuation_adapter_hash"]),
            ]:
                metric = row["evaluation"][when]
                if state in self.evaluations and self.evaluations[state] != metric:
                    raise RuntimeError("Conflicting cached evaluations")
                self.evaluations[state] = metric
            cpool = json.loads((branch / "continuation_pool.json").read_text())
            self.pools[(row["candidate_adapter_hash"], self.seed + 600_000)] = (
                cpool,
                branch / "continuation_pool.json",
            )
            self.trains[key] = (
                branch / "t1",
                row["candidate_adapter_hash"],
                row["runtime"]["candidate"],
                branch / "training_t1.jsonl",
            )
            if row["continuation_available"]:
                self.trains[
                    (
                        row["candidate_adapter_hash"],
                        row["continuation"]["batch_hash"],
                        self.seed + 600_000,
                        50,
                    )
                ] = (
                    branch / "t2",
                    row["continuation_adapter_hash"],
                    row["runtime"]["continuation"],
                    branch / "training_t2.jsonl",
                )

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def note(self, kind, source=None, **details):
        append_jsonl(
            self.receipts,
            {"kind": kind, "source": str(source) if source else None, **details},
        )

    def train(self, batch, seed, max_steps=None, log_path=None):
        key = (
            self.state_hash(),
            batch_hash(batch),
            seed,
            max_steps or self.cfg.training.max_steps,
        )
        cached = self.trains.get(key)
        if cached:
            checkpoint, expected, log, original_log = cached
            started = time.perf_counter()
            self.backend.load_checkpoint(checkpoint)
            if self.state_hash() != expected:
                raise RuntimeError("Cached training checkpoint hash mismatch")
            if log_path and original_log.exists():
                shutil.copyfile(original_log, log_path)
            result = deepcopy(log)
            if result.get("reused_partial_t1"):
                result["seconds"] = time.perf_counter() - started
                self.note(
                    "partial_t1_checkpoint_reused",
                    checkpoint,
                    adapter_hash=expected,
                    source_training_log_sha256=result["source_training_log_sha256"],
                    completed_optimizer_steps=50,
                    new_optimizer_steps=0,
                )
            else:
                self.note("completed_update_reused", checkpoint, adapter_hash=expected)
            return result
        result = self.backend.train(batch, seed, max_steps=max_steps, log_path=log_path)
        if key in self.replay_checks:
            if self.state_hash() != self.replay_checks[key]:
                raise RuntimeError(
                    "Native replay differs from saved unfinished-branch T1"
                )
            self.note(
                "unfinished_branch_t1_replay_matches", adapter_hash=self.state_hash()
            )
        return result

    @contextmanager
    def hooks(self):
        import third_eye.experiments.labeling as module

        old_collect, old_evaluate = module.collect_corrections, module.evaluate
        original_features = module.extract_features

        def collect(backend, examples, verifier, protocol, seed):
            if protocol != self.cfg.protocol:
                raise RuntimeError("Recovery protocol differs")
            cached = self.pools.get((backend.state_hash(), seed))
            if cached:
                pool, source = cached
                self.note("completed_harvest_reused", source)
                return [
                    Correction(Example(**c["example"]), c["completion"], c["attempt"])
                    for c in pool["items"]
                ], deepcopy(pool["stats"])
            return old_collect(backend, examples, verifier, protocol, seed)

        def evaluate(backend, splits, verifier, tokens, seed):
            if tokens != self.cfg.protocol.max_new_tokens or seed != self.seed:
                raise RuntimeError("Recovery evaluation settings differ")
            cached = self.evaluations.get(backend.state_hash())
            if cached is not None:
                self.note(
                    "completed_evaluation_reused", adapter_hash=backend.state_hash()
                )
                return Consequences(**cached)
            return old_evaluate(backend, splits, verifier, tokens, seed)

        def features(backend, batch, anchor, protocol, seed, log_path):
            cached = self.features.get((backend.state_hash(), batch_hash(batch), seed))
            if cached is not None:
                self.note("completed_features_reused", batch_hash=batch_hash(batch))
                return deepcopy(cached)
            return original_features(backend, batch, anchor, protocol, seed, log_path)

        self.extract_features = features
        module.collect_corrections, module.evaluate = collect, evaluate
        try:
            yield self
        finally:
            module.collect_corrections, module.evaluate = old_collect, old_evaluate
