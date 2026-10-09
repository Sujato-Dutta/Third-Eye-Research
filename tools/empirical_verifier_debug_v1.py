# ruff: noqa: E402
"""Bounded repeated verification of identical code; preserve raw diagnostics."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from empirical_common_v1 import BASE, prepared
from empirical_noise_v1 import frozen
from third_eye.config import Config
from third_eye.cluster import require_gpu_allocation
from third_eye.data.splits import load_manifest
from third_eye.evaluation.benchmarks import make_verifier
from third_eye.evaluation.runner import generate_greedy, verify_completions
from third_eye.evaluation.sandbox import run_bounded
from third_eye.io import digest, write_json


def main():
    frozen()
    require_gpu_allocation()
    from third_eye.training.hf_backend import HFBackend

    task = prepared()["noise_tasks"][0]
    out = BASE / "verifier_debug_v1"
    out.mkdir(exist_ok=False)
    cfg = Config.load(task["config"])
    _, splits = load_manifest(task["manifest"])
    verifier = make_verifier()
    backend = HFBackend(cfg)
    backend.load_checkpoint(BASE / "replay_debug_v1/parent")
    examples = splits["ood_dev"]
    texts = generate_greedy(
        backend,
        examples,
        cfg.protocol.max_new_tokens,
        [cfg.protocol.seed] * len(examples),
    )
    write_json(
        out / "fixed_completions.json",
        [
            dict(
                id=ex.id,
                prompt_sha256=ex.prompt_hash,
                completion=text,
                prediction_sha256=digest(text),
            )
            for ex, text in zip(examples, texts)
        ],
    )
    repeats = []
    for i in range(8):
        verdicts = verify_completions(verifier, examples, texts)
        repeats.append([int(v) for v in verdicts])
        write_json(out / "repeated_verdicts.json", repeats)
        print(
            json.dumps(dict(repeat=i, passed=sum(verdicts), count=len(examples))),
            flush=True,
        )
    fluctuations = []
    for j, ex in enumerate(examples):
        bits = [r[j] for r in repeats]
        if len(set(bits)) > 1:
            fluctuations.append(
                dict(
                    id=ex.id,
                    verdicts=bits,
                    example_metadata=ex.metadata,
                    completion=texts[j],
                )
            )
    flags = []
    code = "import sys,os,json; print(json.dumps(dict(isolated=sys.flags.isolated,ignore_environment=sys.flags.ignore_environment,env_seed=os.environ.get('PYTHONHASHSEED'),hash_value=hash('third-eye-verifier'),set_order=list({'alpha','beta','gamma','delta'}))))"
    for i in range(8):
        r = run_bounded(
            verifier.sandbox.command(None, ["-c", code]), 10, kill_group=True
        )
        if r.returncode:
            raise RuntimeError("Sandbox flag diagnostic failed")
        flags.append(json.loads(r.stdout))
    write_json(
        out / "completed.json",
        dict(
            status="verifier_repeatability_review_required",
            ood_correct_counts=[sum(r) for r in repeats],
            fluctuating_items=fluctuations,
            sandbox_flags=flags,
            training_changed=False,
            original_labels_changed=False,
            verifier_flags_changed=False,
        ),
    )
    print(
        json.dumps(
            dict(
                fluctuating_items=len(fluctuations),
                hash_values_distinct=len({r["hash_value"] for r in flags}),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
