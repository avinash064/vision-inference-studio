from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import DownloadedResource, ExternalProvider, directory_bytes, run_with_progress
from ..schemas import AuthState, ExternalFile, ExternalResource, ExternalResourceRef, ProviderCapabilities, ResourceKind, SearchPage


def _value(item: Any, name: str, default=None):
    return getattr(item, name, default) if not isinstance(item, dict) else item.get(name, default)


class _TransferTracker:
    """Byte progress across sequential hf_hub_download calls, fed by huggingface_hub's tqdm_class hook.

    Xet downloads open two bars per file (network transfer and file reconstruction); taking the
    furthest bar for the current file avoids double counting.
    """

    def __init__(self) -> None:
        import threading
        from huggingface_hub.utils import tqdm as hf_tqdm

        self.lock = threading.Lock()
        self.finished = 0
        self.current_size = 0
        self.bars: list[Any] = []
        tracker = self

        class Bar(hf_tqdm):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = True  # silent: the backend log is not a terminal
                super().__init__(*args, **kwargs)
                self.seen = 0
                with tracker.lock:
                    tracker.bars.append(self)

            def update(self, n=1):
                self.seen += n or 0
                return super().update(n)

        self.bar_class = Bar

    def start_file(self, size: int | None) -> None:
        with self.lock:
            self.bars, self.current_size = [], size or 0

    def finish_file(self, size: int) -> None:
        with self.lock:
            self.finished += size
            self.bars, self.current_size = [], 0

    def bytes_done(self) -> int:
        with self.lock:
            current = max((bar.seen for bar in self.bars), default=0)
            return self.finished + (min(current, self.current_size) if self.current_size else current)


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
        from huggingface_hub import hf_hub_download, snapshot_download
        details = self.details(ref)
        destination.mkdir(parents=True, exist_ok=True)
        repo_type = "dataset" if ref.kind == ResourceKind.DATASET else "model"
        revision = details.resolved_revision or ref.revision or ref.version
        files = [file for file in details.files if file.name]
        tracker = _TransferTracker()

        def work() -> None:
            if not files:  # listing unavailable: fall back to one snapshot call
                snapshot_download(repo_id=ref.resource_id, repo_type=repo_type, revision=revision, local_dir=str(destination), token=self.token)
                return
            # File by file so a cancelled job stops after the current file instead of the whole repo.
            for file in files:
                if cancelled():
                    return
                tracker.start_file(file.size_bytes)
                local = hf_hub_download(
                    repo_id=ref.resource_id, filename=file.name, repo_type=repo_type, revision=revision,
                    local_dir=str(destination), token=self.token, tqdm_class=tracker.bar_class,
                )
                tracker.finish_file(Path(local).stat().st_size)

        measure = tracker.bytes_done if files else None
        run_with_progress(work, destination, details.size_bytes, progress, cancelled, measure=measure)
        if cancelled():
            raise InterruptedError("Download cancelled")
        size = directory_bytes(destination)
        progress(size, size)
        return DownloadedResource(destination, details.resolved_revision)
