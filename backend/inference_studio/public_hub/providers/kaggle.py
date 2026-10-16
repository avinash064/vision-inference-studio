from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import DownloadedResource, ExternalProvider, directory_bytes, run_with_progress
from ..schemas import AuthState, ExternalFile, ExternalResource, ExternalResourceRef, ProviderCapabilities, ResourceKind, SearchPage


def _get(item: Any, *names: str, default=None):
    for name in names:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
        if value is not None:
            return value
    return default


class KaggleProvider(ExternalProvider):
    name = "kaggle"
    display_name = "Kaggle"
    capabilities = ProviderCapabilities(search_datasets=True, download_datasets=True, search_models=True, download_models=True, revisions=True)

    def _configured(self) -> bool:
        return bool(os.getenv("KAGGLE_API_TOKEN") or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")).joinpath("kaggle.json").is_file())

    def _api(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api

    def auth_state(self) -> AuthState:
        return AuthState.CONFIGURED if self._configured() else AuthState.NOT_CONFIGURED

    def test_auth(self) -> AuthState:
        if not self._configured():
            return AuthState.NOT_CONFIGURED
        try:
            self._api().dataset_list(page=1)
            return AuthState.CONFIGURED
        except Exception:
            return AuthState.AUTH_FAILED

    def search(self, kind: ResourceKind, query: str, page: int, page_size: int) -> SearchPage:
        api = self._api()
        values = api.dataset_list(search=query or None, page=page) if kind == ResourceKind.DATASET else api.model_list(search=query or None, page_size=page_size)
        items = [self._summary(item, kind) for item in (values or [])[:page_size]]
        return SearchPage(items=items, page=page, page_size=page_size, has_more=len(values or []) >= page_size)

    def _summary(self, item: Any, kind: ResourceKind) -> ExternalResource:
        rid = str(_get(item, "ref", "id", "slug", default=""))
        owner = _get(item, "owner_name", "ownerRef", "owner")
        title = str(_get(item, "title", "name", default=rid.split("/")[-1]))
        version = _get(item, "current_version_number", "versionNumber", "version")
        size = _get(item, "total_bytes", "totalBytes", "size")
        preview = str(_get(item, "thumbnail_image_url", "thumbnailImageUrl", default="") or "")
        if not preview.startswith("https://storage.googleapis.com/kaggle-datasets-images/"):
            preview = ""
        return ExternalResource(
            ref=ExternalResourceRef(provider=self.name, kind=kind, resource_id=rid, version=str(version) if version is not None else None),
            title=title, owner=str(owner) if owner else None, description=str(_get(item, "subtitle", default="") or ""),
            size_bytes=int(size) if size else None, license=_get(item, "license_name", "licenseName"),
            source_url=f"https://www.kaggle.com/{'datasets' if kind == ResourceKind.DATASET else 'models'}/{rid}",
            preview_url=preview or None if kind == ResourceKind.DATASET else None,
        )

    def details(self, ref: ExternalResourceRef) -> ExternalResource:
        api = self._api()
        item = api.dataset_view(ref.resource_id) if ref.kind == ResourceKind.DATASET else api.model_get(ref.resource_id)
        result = self._summary(item, ref.kind)
        result.ref = ref
        files = _get(item, "files", default=[]) or []
        result.files = [ExternalFile(name=str(_get(file, "name", "ref", default="")), size_bytes=_get(file, "total_bytes", "totalBytes", "size")) for file in files]
        result.resolved_revision = str(_get(item, "current_version_number", "versionNumber", default=ref.version) or "") or None
        return result

    def download(self, ref: ExternalResourceRef, destination: Path, requested_format: str | None, progress, cancelled) -> DownloadedResource:
        if cancelled():
            raise InterruptedError("Download cancelled")
        api = self._api()
        destination.mkdir(parents=True, exist_ok=True)
        if ref.kind == ResourceKind.DATASET:
            work = lambda: api.dataset_download_files(ref.resource_id, path=str(destination), force=True, quiet=True, unzip=False)
        else:
            identity = ref.resource_id
            if ref.version and identity.count("/") >= 3 and not identity.endswith(f"/{ref.version}"):
                identity += f"/{ref.version}"
            if identity.count("/") < 3:
                raise ValueError("Kaggle model download requires owner/model/framework/variation and optional version")
            work = lambda: api.model_instance_version_download(identity, path=str(destination), force=True, quiet=True, untar=False)
        # The SDK call is a single blocking archive download; report bytes as they land (total unknown).
        run_with_progress(work, destination, None, progress, cancelled)
        if cancelled():
            raise InterruptedError("Download cancelled")
        files = [path for path in destination.rglob("*") if path.is_file()]
        if not files:
            raise RuntimeError("Kaggle returned no downloadable files")
        total = sum(path.stat().st_size for path in files)
        progress(total, total)
        return DownloadedResource(destination, ref.version)
