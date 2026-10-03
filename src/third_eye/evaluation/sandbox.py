"""Code execution in containers or Linux namespaces; no host fallback."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
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


def run_bounded(command, timeout, limit=65536, kill_group=False):
    """Bound host-side output memory as well as container compute resources."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=kill_group,
    )
    buffers = [bytearray(), bytearray()]
    overflow = threading.Event()

    def kill():
        try:
            if kill_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass

    def read(stream, index):
        while chunk := stream.read(4096):
            if len(buffers[index]) + len(chunk) > limit:
                overflow.set()
                kill()
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
        kill()
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


def code_and_tests(example, completion):
    code, meta = extract_code(completion), example.metadata
    if meta.get("benchmark") == "humaneval":
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
    return code, meta.get("test_setup_code", "") + "\n" + tests


def test_runner(marker):
    return (
        "import runpy\n"
        "namespace = runpy.run_path('/input/candidate.py')\n"
        "namespace.pop('__builtins__', None)\n"
        "test_code = open('/input/tests.py').read()\n"
        "exec(compile(test_code, '/input/tests.py', 'exec'), namespace)\n"
        f"print({marker!r})\n"
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
        code, tests = code_and_tests(example, completion)
        name = "third-eye-" + uuid.uuid4().hex
        # Candidate and tests have separate files. A per-execution nonce avoids
        # accepting ordinary printed text or an early successful process exit.
        marker = "THIRD_EYE_TESTS_PASSED_" + uuid.uuid4().hex
        runner = test_runner(marker)
        with tempfile.TemporaryDirectory(prefix="third_eye_code_") as directory:
            path = Path(directory)
            (path / "candidate.py").write_text(code, encoding="utf-8")
            (path / "tests.py").write_text(tests, encoding="utf-8")
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


class BubblewrapSandbox:
    """Read-only Python runtime in separate user/PID/network/mount namespaces.

    Kernel rlimits bound address space, CPU, files and processes. This requires
    functional unprivileged namespaces and never downgrades to host execution.
    """

    def __init__(
        self,
        executable="bwrap",
        runtime=None,
        timeout=8,
        memory_mb=1024,
        executable_sha256=None,
    ):
        if os.name != "posix" or timeout <= 0 or memory_mb < 128:
            raise SandboxUnavailable("Bubblewrap requires Linux and valid limits")
        binary = shutil.which(executable)
        if not binary:
            raise SandboxUnavailable("Bubblewrap executable unavailable")
        self.executable = str(Path(binary).resolve())
        if (
            executable_sha256
            and hashlib.sha256(Path(binary).read_bytes()).hexdigest()
            != executable_sha256
        ):
            raise SandboxUnavailable("Bubblewrap executable digest mismatch")
        self.runtime = Path(runtime).resolve() if runtime else None
        if self.runtime and not (self.runtime / "bin/python").exists():
            raise SandboxUnavailable("Sandbox Python runtime unavailable")
        self.timeout, self.memory_mb = timeout, memory_mb
        try:
            result = run_bounded(
                self.command(
                    None,
                    [
                        "-c",
                        "from pathlib import Path; assert not Path('/dgxa_home').exists(); assert not Path('/home').exists(); print('READY')",
                    ],
                ),
                10,
                kill_group=True,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable("Bubblewrap namespace startup failed") from exc
        if result.returncode or result.stdout.strip() != b"READY":
            raise SandboxUnavailable(
                "Bubblewrap namespace startup failed: "
                + result.stderr.decode(errors="replace")[:300]
            )

    def command(self, directory, arguments):
        argv = [
            self.executable,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
            "--symlink",
            "usr/bin",
            "/bin",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/tmp",
            "--uid",
            "65534",
            "--gid",
            "65534",
        ]
        if Path("/lib64").exists():
            argv.extend(("--ro-bind", "/lib64", "/lib64"))
        for key, value in {
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "TMPDIR": "/tmp",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "LANG": "C.UTF-8",
        }.items():
            argv.extend(("--setenv", key, value))
        python = "/usr/bin/python3"
        if self.runtime:
            argv.extend(("--ro-bind", str(self.runtime), "/runtime"))
            python = "/runtime/bin/python"
        if directory:
            argv.extend(("--ro-bind", str(Path(directory).resolve()), "/input"))
        return [*argv, python, "-I", "-B", *arguments]

    def verify(self, example, completion):
        code, tests = code_and_tests(example, completion)
        marker = "THIRD_EYE_TESTS_PASSED_" + uuid.uuid4().hex
        limits = (
            "import resource\n"
            f"resource.setrlimit(resource.RLIMIT_AS, ({self.memory_mb * 1024**2},) * 2)\n"
            "resource.setrlimit(resource.RLIMIT_CPU, (4, 4))\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))\n"
            "resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))\n"
            "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n"
            "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        )
        with tempfile.TemporaryDirectory(prefix="third_eye_code_") as directory:
            path = Path(directory)
            for name, content in {
                "candidate.py": code,
                "tests.py": tests,
                "runner.py": limits + test_runner(marker),
            }.items():
                (path / name).write_text(content, encoding="utf-8")
                (path / name).chmod(0o644)
            path.chmod(0o755)
            try:
                result = run_bounded(
                    self.command(path, ["/input/runner.py"]),
                    self.timeout,
                    kill_group=True,
                )
            except subprocess.TimeoutExpired:
                return False
            if result.returncode and result.stderr.startswith(b"bwrap:"):
                raise SandboxUnavailable("Bubblewrap execution infrastructure failed")
            return result.returncode == 0 and result.stdout.rstrip().endswith(
                marker.encode()
            )


def sandbox_from_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("engine") == "bubblewrap":
        return BubblewrapSandbox(
            config.get("executable", "bwrap"),
            config.get("runtime"),
            config.get("timeout_seconds", 8),
            config.get("memory_mb", 1024),
            config.get("executable_sha256"),
        )
    if config.get("engine") != "docker":
        raise ValueError("Select an isolated Docker or Bubblewrap verifier")
    return DockerSandbox(
        config["image"], config.get("timeout_seconds", 8), config.get("memory_mb", 256)
    )
