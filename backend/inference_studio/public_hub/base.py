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
