"""Generated requests must target exactly the validated relay allowlist."""

import empirical_checkpoint_path_repair_v1 as repair


def test_all_generated_launcher_paths_match_current_relay():
    for mode in ("trajectory", "noise", "confirm", "review"):
        index = 7 if mode in ("trajectory", "noise") else None
        args = repair.arguments(mode, index, [123, 456])
        script = next(a for a in args if a.endswith(".slurm"))
        assert script.endswith(
            f"empirical_checkpoint_{'gpu' if index is not None else 'cpu'}_v3.slurm"
        )
        assert "--dependency=afterany:123:456" in args


def test_repair_changes_only_the_script_filename():
    for mode in ("trajectory", "noise", "confirm", "review"):
        index = 2 if mode in ("trajectory", "noise") else None
        old = repair.ORIGINAL(mode, index)
        new = repair.arguments(mode, index)
        differences = [(a, b) for a, b in zip(old, new) if a != b]
        assert len(differences) == 1
        assert differences[0][1] == differences[0][0].replace("_v2.slurm", "_v3.slurm")
