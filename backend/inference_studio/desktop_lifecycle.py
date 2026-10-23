"""Backend lifecycle when launched by the desktop shell: parent watchdog and graceful self-shutdown.

Windows needs its own code paths: ``os.kill(pid, 0)`` is a harmless existence probe on POSIX but calls
``TerminateProcess`` on Windows (it would kill the Electron shell), and a SIGTERM sent with ``os.kill`` there
is an abrupt termination that skips uvicorn's shutdown hooks (model unload, GPU release).
"""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Callable

IS_WINDOWS = os.name == "nt"


def process_create_time(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:
        return None


def parent_alive(pid: int, expected_create_time: float | None = None) -> bool:
    """Whether ``pid`` still refers to the process that started us."""
    if IS_WINDOWS:
        try:
            import psutil
        except ImportError:  # psutil is a core requirement; stay up rather than guess
            return True
        if not psutil.pid_exists(pid):
            return False
        # Windows recycles PIDs; a different creation time means the shell exited and the PID was reused.
        current = process_create_time(pid)
        return expected_create_time is None or current is None or abs(current - expected_create_time) < 1.0
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def request_graceful_shutdown() -> None:
    """Ask the running uvicorn server to exit through its normal shutdown path."""
    if IS_WINDOWS:
        # Delivered to uvicorn's Python-level SIGTERM handler on the main thread (not TerminateProcess).
        signal.raise_signal(signal.SIGTERM)
    else:
        os.kill(os.getpid(), signal.SIGTERM)


def watch_parent_process(
    parent_pid: int,
    interval: float = 2.0,
    alive: Callable[[int, float | None], bool] = parent_alive,
    shutdown: Callable[[], None] = request_graceful_shutdown,
) -> threading.Thread:
    """Shut down when the desktop shell that started this backend is gone.

    The shell stops the backend on a clean quit, but a crash, a hard kill or a signal handled by Chromium skips
    that, leaving an orphaned backend holding loaded models (GBs of GPU memory)."""
    # On POSIX, being re-parented also means the direct parent died (immune to zombies/PID reuse).
    direct_child = not IS_WINDOWS and os.getppid() == parent_pid
    expected = process_create_time(parent_pid)

    def run() -> None:
        while True:
            time.sleep(interval)
            gone = not alive(parent_pid, expected) or (direct_child and os.getppid() != parent_pid)
            if gone:
                shutdown()
                return

    thread = threading.Thread(target=run, name="parent-watchdog", daemon=True)
    thread.start()
    return thread
