from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .schemas import AuthState, ExternalResource, ExternalResourceRef, ProviderCapabilities, ResourceKind, SearchPage


ProgressCallback = Callable[[int, int | None], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class DownloadedResource:
    path: Path
    resolved_revision: str | None = None
    provider_checksum: str | None = None


def directory_bytes(path: Path) -> int:
    """Bytes currently on disk below path, including SDK partial/.incomplete files."""
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:  # file renamed or removed by the SDK mid-scan
            continue
    return total


def run_with_progress(
    work: Callable[[], object],
    destination: Path,
    total: int | None,
    progress: ProgressCallback,
    cancelled: CancelCallback,
    interval: float = 0.5,
    measure: Callable[[], int] | None = None,
) -> object:
    """Run a blocking SDK download in a worker thread while reporting on-disk progress.

    By default progress is measured from what has landed in destination; pass measure when the
    SDK offers a better byte count. Cancellation returns immediately; the abandoned worker
    finishes its current file and exits (work should check cancelled() between files).
    """
    measure = measure or (lambda: directory_bytes(destination))
    import threading

    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["value"] = work()
        except BaseException as exc:  # re-raised in the caller's thread
            outcome["error"] = exc

    worker = threading.Thread(target=target, name="public-hub-transfer", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(interval)
        if cancelled():
            raise InterruptedError("Download cancelled")
        done = measure()
        progress(min(done, total) if total else done, total)
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome.get("value")


class ExternalProvider(ABC):
    name: str
    display_name: str
    capabilities: ProviderCapabilities

    @abstractmethod
    def auth_state(self) -> AuthState: ...

    @abstractmethod
    def test_auth(self) -> AuthState: ...

    @abstractmethod
    def search(self, kind: ResourceKind, query: str, page: int, page_size: int) -> SearchPage: ...

    @abstractmethod
    def details(self, ref: ExternalResourceRef) -> ExternalResource: ...

    @abstractmethod
    def download(self, ref: ExternalResourceRef, destination: Path, requested_format: str | None,
                 progress: ProgressCallback, cancelled: CancelCallback) -> DownloadedResource: ...

    def public_status(self) -> dict:
        return {
            "id": self.name,
            "display_name": self.display_name,
            "auth_state": self.auth_state().value,
            "capabilities": self.capabilities.model_dump(mode="json"),
        }
