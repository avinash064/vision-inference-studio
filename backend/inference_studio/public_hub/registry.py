from __future__ import annotations

from .base import ExternalProvider
from .schemas import ResourceKind
from .security import redact_error


class ExternalProviderRegistry:
    def __init__(self, providers: list[ExternalProvider] | None = None) -> None:
        self._providers: dict[str, ExternalProvider] = {}
        for provider in providers or []:
            self.register(provider)

    def register(self, provider: ExternalProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")
        self._providers[provider.name] = provider

    def get(self, name: str) -> ExternalProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ValueError(f"Unknown public resource provider: {name}") from exc

    def list(self) -> list[dict]:
        return [provider.public_status() for provider in self._providers.values()]

    def search(self, kind: ResourceKind, query: str, provider: str | None, page: int, page_size: int) -> dict:
        selected = [self.get(provider)] if provider else list(self._providers.values())
        items, failures = [], []
        for implementation in selected:
            capability = implementation.capabilities.search_datasets if kind == ResourceKind.DATASET else implementation.capabilities.search_models
            if not capability:
                continue
            try:
                items.extend(implementation.search(kind, query, page, page_size).items)
            except Exception as exc:
                failures.append({"provider": implementation.name, "error": redact_error(exc)})
        return {"items": [item.model_dump(mode="json") for item in items[:page_size]], "page": page, "page_size": page_size, "has_more": len(items) > page_size, "provider_errors": failures}
