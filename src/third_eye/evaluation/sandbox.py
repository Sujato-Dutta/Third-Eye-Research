"""Code execution in an OCI container; no host-execution fallback."""

import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import textwrap
import uuid
from types import SimpleNamespace


def extract_code(completion):
    completion = completion.split("</think>")[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.S)
    return max(blocks, key=len).strip() if blocks else completion.strip()


class SandboxUnavailable(RuntimeError):
    pass


def run_bounded(command, timeout, limit=65536):
    """Bound host-side output memory as well as container compute resources."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffers = [bytearray(), bytearray()]
    overflow = threading.Event()

    def read(stream, index):
        while chunk := stream.read(4096):
            if len(buffers[index]) + len(chunk) > limit:
                overflow.set()
                process.kill()
                break
            buffers[index].extend(chunk)

    readers = [
        threading.Thread(target=read, args=(stream, i), daemon=True)
        for i, stream in enumerate((process.stdout, process.stderr))
    ]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5)
    if overflow.is_set():
        raise subprocess.TimeoutExpired(command, timeout)
    return SimpleNamespace(
        returncode=process.returncode,
        stdout=bytes(buffers[0]),
        stderr=bytes(buffers[1]),
    )


class DockerSandbox:
    def __init__(self, image, timeout=8, memory_mb=256):
        if not image or "@sha256:" not in image:
            raise ValueError("Pin the code sandbox image by OCI digest")
        if timeout <= 0 or memory_mb < 64:
            raise ValueError("Invalid sandbox resource budget")
        self.image, self.timeout, self.memory_mb = image, timeout, memory_mb
        try:
            check = subprocess.run(
                ["docker", "image", "inspect", image], capture_output=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(
                "An operational Docker sandbox is required"
            ) from exc
        if check.returncode:
            raise SandboxUnavailable(
                "Pre-pull the pinned sandbox image on the execution host"
            )

    def verify(self, example, completion):
        code = extract_code(completion)
        meta = example.metadata
        if meta.get("benchmark") == "humaneval":
            # HumanEval expects continuation of its function prefix; accept full
            # function solutions too, but never add tests to the model prompt.
            entry = meta["entry_point"]
            if not re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, re.M):
                code = (
                    meta["code_prefix"].rstrip()
                    + "\n"
                    + textwrap.indent(textwrap.dedent(code), "    ")
                )
            tests = meta["test"] + f"\ncheck({entry})\n"
        else:
            tests = "\n".join(meta.get("test_list", []))
            if not tests:
                raise ValueError("Code example is missing verifier tests")
        setup = meta.get("test_setup_code", "")
        name = "third-eye-" + uuid.uuid4().hex
        # Candidate and tests have separate files. A per-execution nonce avoids
        # accepting ordinary printed text or an early successful process exit.
        marker = "THIRD_EYE_TESTS_PASSED_" + uuid.uuid4().hex
        runner = (
            "import runpy\n"
            "namespace = runpy.run_path('/input/candidate.py')\n"
            "namespace.pop('__builtins__', None)\n"
            "test_code = open('/input/tests.py').read()\n"
            "exec(compile(test_code, '/input/tests.py', 'exec'), namespace)\n"
            f"print({marker!r})\n"
        )
        with tempfile.TemporaryDirectory(prefix="third_eye_code_") as directory:
            path = Path(directory)
            (path / "candidate.py").write_text(code, encoding="utf-8")
            (path / "tests.py").write_text(setup + "\n" + tests, encoding="utf-8")
            (path / "runner.py").write_text(runner, encoding="utf-8")
            path.chmod(0o755)
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=32",
                "--ulimit=cpu=4:4",
                "--ulimit=fsize=1048576:1048576",
                "--log-driver=none",
                f"--memory={self.memory_mb}m",
                "--cpus=1",
                "--user=65534:65534",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
                "--mount",
                f"type=bind,source={path.resolve()},target=/input,readonly",
                self.image,
                "python",
                "-I",
                "-B",
                "/input/runner.py",
            ]
            try:
                result = run_bounded(command, self.timeout)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["docker", "rm", "-f", name], capture_output=True, timeout=10
                )
                return False
            if result.returncode in (125, 126, 127):
                raise SandboxUnavailable(
                    "Code container failed to start: "
                    + result.stderr.decode(errors="replace")[:300]
                )
            return result.returncode == 0 and result.stdout.rstrip().endswith(
                marker.encode()
            )


def sandbox_from_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("engine") != "docker":
        raise ValueError("Only the isolated Docker verifier is supported")
    return DockerSandbox(
        config["image"], config.get("timeout_seconds", 8), config.get("memory_mb", 256)
    )
