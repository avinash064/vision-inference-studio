"""Smoke test for a packaged Windows build (run by CI on windows-latest; no GPU needed).

  python packaging/windows/smoke_test.py --app-dir "desktop/out/Vision Inference Studio-win32-x64" --work <tempdir>

1. The bundled CPython installs a private runtime (core profile) with packaging/runtime/setup_runtime.py.
2. The backend starts the way Electron starts it, with VIS_PARENT_PID set to *this* process: the old
   os.kill(pid, 0) watchdog would terminate this script on Windows, so surviving proves the fix.
3. /health and /api/v1/runtime/diagnostics answer; POST /api/v1/desktop/shutdown exits gracefully.
4. The packaged Electron executable starts, launches its backend and reports "Backend ready".
Writes <work>/smoke-report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def request(url: str, token: str, method: str = "GET") -> tuple[int, dict]:
    call = urllib.request.Request(url, method=method, headers={"X-Vision-Session": token})
    try:
        with urllib.request.urlopen(call, timeout=60) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, {}
    except OSError:  # refused/reset: the backend is not listening (exited or never started)
        return 0, {}


def wait_for(predicate, timeout: float, interval: float = 0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--skip-electron", action="store_true")
    args = parser.parse_args()
    args.app_dir = args.app_dir.resolve()  # absolute: the backend runs with cwd=resources, like Electron starts it
    resources = args.app_dir / "resources"
    work = args.work.resolve()
    data, config = work / "Local Data ü" / "Vision Inference Studio", work / "Roaming ü" / "Vision Inference Studio"
    report: dict = {"steps": {}}

    # 1. Managed runtime from the bundled interpreter.
    bundled = resources / "python" / ("python.exe" if os.name == "nt" else "bin/python3.12")
    started = time.monotonic()
    setup = subprocess.run([str(bundled), "-X", "utf8", "-I", str(resources / "packaging" / "runtime" / "setup_runtime.py"), "--mode", "install", "--profile", "core", "--app-root", str(resources), "--data-root", str(data), "--app-version", "smoke"], capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    (work / "runtime-setup.out").write_text(setup.stdout + setup.stderr, encoding="utf-8")
    runtime_python = data / "runtime" / "python" / ("python.exe" if os.name == "nt" else "bin/python3.12")
    manifest = json.loads((data / "runtime" / "runtime-manifest.json").read_text(encoding="utf-8")) if setup.returncode == 0 else {}
    report["steps"]["runtime_install"] = {"ok": setup.returncode == 0 and runtime_python.is_file(), "exit_code": setup.returncode, "seconds": round(time.monotonic() - started, 1), "manifest": {key: manifest.get(key) for key in ("platform", "arch", "python", "runtime_version", "profile")}}
    if not report["steps"]["runtime_install"]["ok"]:
        report["steps"]["runtime_install"]["output_tail"] = (setup.stdout + setup.stderr)[-4000:]
    if not report["steps"]["runtime_install"]["ok"]:
        return finish(work, report)

    # 2-3. Backend launched like the Electron shell does.
    token, port = secrets.token_hex(16), free_port()
    env = {**os.environ, "PYTHONPATH": str(resources / "backend"), "PYTHONUTF8": "1", "VIS_DESKTOP_MODE": "1", "VIS_API_TOKEN": token, "VIS_SHELL_TOKEN": secrets.token_hex(16),
           "VIS_PARENT_PID": str(os.getpid()), "VIS_RUNTIME_SOURCE": "managed", "VIS_RUNTIME_DIR": str(data / "runtime"), "VIS_PROJECT_ROOT": str(resources), "VIS_DATA_DIR": str(data),
           "VIS_CONFIG_DIR": str(config), "VIS_MODELS_FILE": str(resources / "config" / "models.yaml"), "VIS_ENV_FILE": str(config / ".env"), "VIS_LOG_DIR": str(data / "logs")}
    backend_log = (work / "backend.out").open("w", encoding="utf-8")
    backend = subprocess.Popen([str(runtime_python), "-m", "uvicorn", "inference_studio.main:app", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"], cwd=resources, env=env, stdout=backend_log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    base = f"http://127.0.0.1:{port}"
    healthy = wait_for(lambda: backend.poll() is None and _ok(lambda: request(f"{base}/health", token)[0] == 200), 90)
    time.sleep(6)  # > 2 watchdog intervals: on the old code this process would already have been terminated
    status, diagnostics = request(f"{base}/api/v1/runtime/diagnostics", token)
    report["steps"]["backend"] = {
        "ok": bool(healthy) and status == 200 and backend.poll() is None,
        "watchdog_left_parent_alive": True,  # reaching this line is the proof
        "diagnostics": {key: diagnostics.get(key) for key in ("platform", "architecture", "python", "runtime_source", "python_executable", "cuda_status", "cuda_detail", "runtime_manifest")},
        "families": {name: value.get("status") for name, value in diagnostics.get("families", {}).items()},
        "unauthenticated_status": request(f"{base}/health", "wrong-token")[0],
        "healthy_within_90s": bool(healthy),
        "backend_exit_code": backend.poll(),
    }
    if not report["steps"]["backend"]["ok"]:
        backend_log.flush()
        report["steps"]["backend"]["log_tail"] = (work / "backend.out").read_text(encoding="utf-8", errors="replace")[-4000:]
        backend.kill()
        backend_log.close()
        return finish(work, report)
    shutdown_status, _ = request(f"{base}/api/v1/desktop/shutdown", token, method="POST")
    try:
        code = backend.wait(timeout=30)
    except subprocess.TimeoutExpired:
        backend.kill()
        code = None
    backend_log.close()
    log_text = (work / "backend.out").read_text(encoding="utf-8", errors="replace")
    # uvicorn re-raises the signal after a clean shutdown, so the exit code is platform-specific; the
    # lifespan log lines prove models were released through the normal path (not TerminateProcess).
    graceful = "Application shutdown complete" in log_text and "Finished server process" in log_text
    report["steps"]["graceful_shutdown"] = {"ok": shutdown_status == 202 and code is not None and graceful, "http_status": shutdown_status, "exit_code": code, "log_tail": log_text[-1500:]}

    # 4. The packaged Electron app with this runtime (VIS_PYTHON) and isolated profile folders.
    if not args.skip_electron and os.name == "nt":
        exe = args.app_dir / "vision-inference-studio.exe"
        app_env = {**os.environ, "APPDATA": str(config.parent), "LOCALAPPDATA": str(data.parent), "VIS_PYTHON": str(runtime_python)}
        app = subprocess.Popen([str(exe)], env=app_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        electron_log = data / "logs" / "electron.log"
        ready = wait_for(lambda: electron_log.is_file() and "Backend ready" in electron_log.read_text(encoding="utf-8", errors="replace"), 180, 1.0)
        report["steps"]["electron"] = {"ok": bool(ready), "electron_log_tail": electron_log.read_text(encoding="utf-8", errors="replace")[-2000:] if electron_log.is_file() else "", "backend_log_tail": _tail(data / "logs" / "backend.log")}
        subprocess.run(["taskkill", "/PID", str(app.pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
    return finish(work, report)


def _ok(check) -> bool:
    try:
        return bool(check())
    except OSError:
        return False


def _tail(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[-2000:] if path.is_file() else ""


def finish(work: Path, report: dict) -> int:
    report["ok"] = all(step.get("ok") for step in report["steps"].values())
    (work / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
