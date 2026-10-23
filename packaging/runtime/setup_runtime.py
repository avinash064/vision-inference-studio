"""Cross-platform installer for the Vision Inference Studio private Python runtime.

Run by the CPython interpreter bundled with the desktop app (``resources/python``). The Windows desktop shell
uses it; the Linux ``packaging/linux/setup-runtime.sh`` performs the same steps with a venv.

On Windows the managed runtime is a private copy of the bundled interpreter in
``%LOCALAPPDATA%\\Vision Inference Studio\\runtime\\python`` (not a venv): Squirrel installs every app version
into a new ``app-x.y.z`` folder and removes the old one, which would break a venv's ``home`` on upgrade.

Progress lines (``VIS_PROGRESS <step> <total> <message>``) are parsed by the desktop shell. Uses only the
standard library and never invokes a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIREMENT_FILES = ("requirements.txt", "requirements-runtimes.txt", "requirements-chainner-standard.txt", "requirements-desktop.txt", "requirements-lock.txt", "requirements-public-hub.txt", "pyproject.toml")
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"  # the lock pins CUDA 12.1 builds of torch/torchvision
RUNTIME_LAYOUT_VERSION = 1  # bump when the managed runtime layout changes incompatibly
TOTAL_STEPS = 10
IS_WINDOWS = os.name == "nt"


def requirements_hash(app_root: Path) -> str:
    """Identical to RuntimeManager.requirementsHash() in the desktop shell and setup-runtime.sh."""
    digest = hashlib.sha256()
    for name in REQUIREMENT_FILES:
        digest.update(f"{name}\n".encode())
        digest.update((app_root / name).read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def python_in(root: Path) -> Path:
    """The interpreter inside a standalone CPython tree (packaged Linux trees have no python3 symlink)."""
    if IS_WINDOWS:
        return root / "python.exe"
    candidates = sorted(root.glob("bin/python3.[0-9]*"))
    return next((path for path in candidates if path.name.count(".") == 1), root / "bin" / "python3")


def runtime_python(runtime_root: Path) -> Path:
    return python_in(runtime_root / "python")


def interpreter_root(executable: Path) -> Path:
    """Installation root of a python-build-standalone interpreter (python.exe sits at the root on Windows)."""
    return executable.parent if IS_WINDOWS else executable.parent.parent


class Setup:
    def __init__(self, args: argparse.Namespace) -> None:
        self.mode, self.profile = args.mode, args.profile
        self.app_root = Path(args.app_root).resolve()
        self.data_root = Path(args.data_root).resolve()
        self.app_version = args.app_version
        self.runtime_root = self.data_root / "runtime"
        self.log_file = self.data_root / "logs" / "runtime-setup.log"
        self.lock = self.runtime_root / "setup.lock"

    # -- output ---------------------------------------------------------------------------------------------
    def log(self, text: str) -> None:
        print(text, flush=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} {text}\n")

    def progress(self, step: int, message: str) -> None:
        self.log(f"VIS_PROGRESS {step} {TOTAL_STEPS} {message}")

    def run(self, command: list[str]) -> None:
        """Run a child process with an argument list (no shell), streaming its output to the log."""
        self.log("$ " + " ".join(command))
        env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        env.pop("PYTHONPATH", None)
        flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="replace", creationflags=flags) as child:
            assert child.stdout is not None
            for line in child.stdout:
                self.log(line.rstrip())
        if child.returncode:
            raise SystemExit(f"Command failed with exit code {child.returncode}: {command[0]} {' '.join(command[1:4])}")

    # -- locking --------------------------------------------------------------------------------------------
    @staticmethod
    def _pid_running(pid: int) -> bool:
        if IS_WINDOWS:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return code.value == 259  # STILL_ACTIVE
        try:
            os.kill(pid, 0)  # existence probe; never used on Windows where it would terminate the process
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def acquire_lock(self) -> None:
        try:
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = int(self.lock.read_text().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and self._pid_running(owner):
                raise SystemExit("Another runtime setup is already running.")
            self.lock.unlink(missing_ok=True)
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))

    # -- steps ----------------------------------------------------------------------------------------------
    def install(self) -> None:
        started = time.monotonic()
        self.progress(1, "Preparing private application directories")
        for name in REQUIREMENT_FILES:
            if not (self.app_root / name).is_file():
                raise SystemExit(f"Missing packaged file: {name}")

        self.progress(2, "Selecting the bundled Python")
        bundled = Path(sys.executable).resolve()
        bundled_version = platform.python_version()
        self.log(f"Bundled interpreter {bundled_version} ({platform.system()} {platform.machine()})")

        self.progress(3, "Preparing the private runtime")
        target_root = self.runtime_root / "python"
        manifest = self._read_json(self.runtime_root / "runtime-manifest.json")
        if self.mode == "reinstall" or (target_root.exists() and (manifest or {}).get("python") != bundled_version) or (manifest or {}).get("runtime_version") != RUNTIME_LAYOUT_VERSION:
            if target_root.exists():
                self.log("Recreating the runtime (reinstall requested, or the bundled Python/layout changed)")
            shutil.rmtree(target_root, ignore_errors=True)
        staging = self.runtime_root / "python.installing"
        fresh = not runtime_python(self.runtime_root).is_file()
        if fresh:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(interpreter_root(bundled), staging, symlinks=False)
            python = python_in(staging)
        else:
            python = runtime_python(self.runtime_root)

        self.progress(4, "Updating Python packaging tools")
        # -X utf8 survives -I (which drops PYTHONUTF8): child output is UTF-8 even for non-ASCII user folders.
        pip = [str(python), "-X", "utf8", "-I", "-m", "pip", "install", "--no-input"]
        self.run([*pip, "--upgrade", "pip", "setuptools", "wheel"])

        self.progress(5, "Installing backend dependencies")
        lock = str(self.app_root / "requirements-lock.txt")
        self.run([*pip, "--only-binary=:all:", "-c", lock, "-r", str(self.app_root / "requirements.txt")])

        if self.profile == "full":
            self.progress(6, "Installing PyTorch (CUDA 12.1), inference runtimes and chaiNNer support")
            self.run([*pip, "--only-binary=:all:", "--extra-index-url", TORCH_INDEX, "-c", lock, "-r", str(self.app_root / "requirements-desktop.txt")])
            self.progress(7, "Installing bundled Grounding DINO and CLIP")
            wheels = sorted((self.app_root / "wheels").glob("*.whl"))
            if wheels:
                self.run([*pip, "--no-index", "--no-deps", *map(str, wheels)])
        else:
            self.progress(6, "Core profile: skipping model runtimes")
            self.progress(7, "Core profile: skipping bundled model wheels")

        self.progress(8, "Installing the Vision Inference Studio backend")
        with tempfile.TemporaryDirectory(prefix="vis-backend-") as temporary:
            source = Path(temporary)
            shutil.copy2(self.app_root / "pyproject.toml", source / "pyproject.toml")
            shutil.copytree(self.app_root / "backend", source / "backend", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            self.run([*pip, "--no-deps", str(source)])

        self.progress(9, "Validating required imports and compute support")
        check = self._runtime_check(python)
        if not check.get("ok"):
            raise SystemExit("Runtime validation failed: " + ", ".join(check.get("failures", [])))
        if check.get("optional_failures"):
            self.log("Optional engines unavailable (their model families will show as unavailable): " + ", ".join(check["optional_failures"]))

        if fresh:
            if target_root.exists():
                shutil.rmtree(target_root)
            os.replace(staging, target_root)
            python = runtime_python(self.runtime_root)

        self.progress(10, "Writing runtime metadata")
        self._write_metadata(python, check, bundled_version, int(time.monotonic() - started))
        self.log(f"RUNTIME READY ({int(time.monotonic() - started)} seconds)")

    def _runtime_check(self, python: Path) -> dict:
        command = [str(python), "-X", "utf8", "-I", "-m", "inference_studio.runtime_check", "--json", "--profile", self.profile]
        self.log("$ " + " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0)
        self.log(result.stderr.strip()[-4000:])
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            raise SystemExit(f"Runtime check produced no result (exit {result.returncode})")

    def _write_metadata(self, python: Path, check: dict, bundled_version: str, seconds: int) -> None:
        frozen = subprocess.run([str(python), "-X", "utf8", "-I", "-m", "pip", "list", "--format=json"], capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0)
        packages = {row["name"].lower(): row["version"] for row in json.loads(frozen.stdout or "[]")}
        now = datetime.now(timezone.utc).isoformat()
        # Read by the desktop shell (RuntimeManager) to decide whether the runtime matches this app build.
        self._write_json(self.runtime_root / "runtime-version.json", {
            "app_version": self.app_version, "requirements_hash": requirements_hash(self.app_root),
            "python_version": check["python"]["version"], "python_executable": check["python"]["executable"],
            "cuda": check["cuda"], "packages": check["packages"], "profile": self.profile, "created_at": now,
            "durations_seconds": {"total": seconds},
        })
        # Portable description of what was installed (shown by runtime diagnostics, attached to releases).
        self._write_json(self.runtime_root / "runtime-manifest.json", {
            "platform": sys.platform, "arch": {"amd64": "x64", "x86_64": "x64"}.get(platform.machine().lower(), platform.machine().lower()),
            "python": bundled_version, "application_version": self.app_version, "runtime_version": RUNTIME_LAYOUT_VERSION,
            "profile": self.profile, "torch": packages.get("torch"), "cuda_runtime": check["cuda"].get("version"),
            "requirements_hash": requirements_hash(self.app_root), "packages": packages, "created_at": now,
        })

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def main(self) -> int:
        if self.mode == "hash":
            print(requirements_hash(self.app_root))
            return 0
        for directory in (self.runtime_root, self.log_file.parent, self.data_root / "weights", self.data_root / "workflows", self.data_root / "workspace", self.data_root / "benchmarks"):
            directory.mkdir(parents=True, exist_ok=True)
        if self.mode == "check":
            return subprocess.call([str(runtime_python(self.runtime_root)), "-X", "utf8", "-I", "-m", "inference_studio.runtime_check", "--json"])
        self.acquire_lock()
        try:
            self.install()
        finally:
            self.lock.unlink(missing_ok=True)
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["install", "repair", "reinstall", "check", "hash"], default="install")
    parser.add_argument("--profile", choices=["full", "core"], default="full", help="core installs only the API server (CI smoke tests)")
    parser.add_argument("--app-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data-root", required=False, default=os.getenv("VIS_DATA_DIR", ""))
    parser.add_argument("--app-version", default=os.getenv("VIS_APP_VERSION", "0.0.0"))
    args = parser.parse_args(argv)
    if args.mode != "hash" and not args.data_root:
        parser.error("--data-root (or VIS_DATA_DIR) is required")
    return args


if __name__ == "__main__":
    # Run with -I (environment ignored, so PYTHONUTF8 has no effect): force UTF-8 output so user names and
    # folders outside the Windows code page (e.g. CJK) cannot crash progress logging.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(Setup(parse_args()).main())
