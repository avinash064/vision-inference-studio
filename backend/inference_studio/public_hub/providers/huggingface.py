from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import DownloadedResource, ExternalProvider
from ..schemas import AuthState, ExternalFile, ExternalResource, ExternalResourceRef, ProviderCapabilities, ResourceKind, SearchPage


def _value(item: Any, name: str, default=None):
    return getattr(item, name, default) if not isinstance(item, dict) else item.get(name, default)


class HuggingFaceProvider(ExternalProvider):
    name = "huggingface"
    display_name = "Hugging Face"
    capabilities = ProviderCapabilities(search_datasets=True, download_datasets=True, search_models=True, download_models=True, revisions=True)

    @property
    def token(self) -> str | None:
        return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def auth_state(self) -> AuthState:
        return AuthState.CONFIGURED if self.token else AuthState.NOT_CONFIGURED

    def test_auth(self) -> AuthState:
        if not self.token:
            return AuthState.NOT_CONFIGURED
        try:
            self._api().whoami()
            return AuthState.CONFIGURED
        except Exception:
            return AuthState.AUTH_FAILED

    def search(self, kind: ResourceKind, query: str, page: int, page_size: int) -> SearchPage:
        api = self._api()
        limit = page * page_size + 1
        values = list(api.list_datasets(search=query or None, limit=limit, full=True) if kind == ResourceKind.DATASET else api.list_models(search=query or None, limit=limit, full=True))
        values = values[(page - 1) * page_size:page * page_size + 1]
        items = [self._summary(value, kind) for value in values[:page_size]]
        return SearchPage(items=items, page=page, page_size=page_size, has_more=len(values) > page_size)

    def _summary(self, item: Any, kind: ResourceKind) -> ExternalResource:
        resource_id = str(_value(item, "id", _value(item, "modelId", "")))
        tags = list(_value(item, "tags", []) or [])
        sha = _value(item, "sha")
        card = _value(item, "card_data", {}) or {}
        license_name = _value(item, "license") or (card.get("license") if isinstance(card, dict) else None)
        return ExternalResource(
            ref=ExternalResourceRef(provider=self.name, kind=kind, resource_id=resource_id, revision=sha),
            title=resource_id.split("/")[-1], owner=resource_id.split("/")[0] if "/" in resource_id else None,
            tags=tags, task=_value(item, "pipeline_tag"), license=license_name,
            source_url=f"https://huggingface.co/{'datasets/' if kind == ResourceKind.DATASET else ''}{resource_id}", resolved_revision=sha,
        )

    def details(self, ref: ExternalResourceRef) -> ExternalResource:
        info = self._api().dataset_info(ref.resource_id, revision=ref.revision or ref.version, files_metadata=True) if ref.kind == ResourceKind.DATASET else self._api().model_info(ref.resource_id, revision=ref.revision or ref.version, files_metadata=True)
        result = self._summary(info, ref.kind)
        siblings = _value(info, "siblings", []) or []
        result.files = [ExternalFile(name=str(_value(f, "rfilename", "")), size_bytes=_value(f, "size")) for f in siblings]
        result.size_bytes = sum(f.size_bytes or 0 for f in result.files) or None
        result.description = str(_value(info, "description", "") or "")
        result.ref = ref
        result.resolved_revision = _value(info, "sha")
        config = _value(info, "config", {}) or {}
        if isinstance(config, dict):
            result.architecture = ", ".join(config.get("architectures", []) or []) or config.get("model_type")
        result.metadata = {"downloads": _value(info, "downloads"), "likes": _value(info, "likes")}
        return result

    def download(self, ref: ExternalResourceRef, destination: Path, requested_format: str | None, progress, cancelled) -> DownloadedResource:
        if cancelled():
            raise InterruptedError("Download cancelled")
        from huggingface_hub import snapshot_download
        details = self.details(ref)
        destination.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(
            repo_id=ref.resource_id,
            repo_type="dataset" if ref.kind == ResourceKind.DATASET else "model",
            revision=details.resolved_revision or ref.revision or ref.version,
            local_dir=str(destination), token=self.token,
        )
        size = sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())
        progress(size, size)
        if cancelled():
            raise InterruptedError("Download cancelled")
        return DownloadedResource(Path(path), details.resolved_revision)
