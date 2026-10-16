from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
import uuid
from pathlib import Path

from inference_studio.config import Settings
from inference_studio.durable import JsonDocumentStore
from inference_studio.models import WeightStore

from .assets import ImportedAssetRegistry
from .inspectors import DatasetInspector, ModelCompatibilityInspector
from .integrity import IntegrityError, ensure_cache_capacity, safe_extract_archive, tree_identity
from .registry import ExternalProviderRegistry
from .schemas import DownloadState, ExternalResourceRef, ResourceDownloadJob, ResourceDownloadRequest, ResourceKind, utc_now
from .security import redact_error


TERMINAL_DOWNLOAD_STATES = {DownloadState.COMPLETED, DownloadState.CANCELLED, DownloadState.ERROR}


class DownloadManager:
    def __init__(self, settings: Settings, providers: ExternalProviderRegistry, assets: ImportedAssetRegistry, weights: WeightStore | None = None) -> None:
        self.settings, self.providers, self.assets = settings, providers, assets
        self.weights = weights
        self.store = JsonDocumentStore(settings.public_hub_root / "metadata" / "downloads", "dl_")
        self.staging = settings.public_hub_root / "downloads"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.dataset_inspector, self.model_inspector = DatasetInspector(), ModelCompatibilityInspector()
        self._semaphore = asyncio.Semaphore(settings.public_hub_max_concurrent_downloads)
        self._tasks: dict[str, asyncio.Task] = {}
        self._active_cache_paths: set[Path] = set()
        # Cancel requests are also kept in memory: progress writes from the transfer thread
        # re-save the job document and must never resurrect a cancelled download.
        self._cancelled: set[str] = set()

    @staticmethod
    def public(job: ResourceDownloadJob) -> dict:
        return job.model_dump(mode="json", exclude={"local_path"})

    def get(self, identity: str) -> ResourceDownloadJob:
        return ResourceDownloadJob.model_validate(self.store.get(identity))

    def list(self) -> list[dict]:
        return [self.public(ResourceDownloadJob.model_validate(raw)) for raw in sorted(self.store.list(), key=lambda value: value.get("created_at", ""), reverse=True)]

    def create(self, request: ResourceDownloadRequest) -> ResourceDownloadJob:
        provider = self.providers.get(request.resource.provider)
        allowed = provider.capabilities.download_datasets if request.resource.kind == ResourceKind.DATASET else provider.capabilities.download_models
        if not allowed:
            raise ValueError(f"{provider.display_name} does not support {request.resource.kind.value} download")
        for raw in self.store.list():
            existing = ResourceDownloadJob.model_validate(raw)
            if existing.resource == request.resource and existing.requested_format == request.format and existing.state not in {DownloadState.CANCELLED, DownloadState.ERROR}:
                return existing
        job = ResourceDownloadJob(download_id=f"dl_{uuid.uuid4().hex}", resource=request.resource, requested_format=request.format)
        self._save(job)
        self._schedule(job.download_id)
        return job

    def _save(self, job: ResourceDownloadJob) -> None:
        job.updated_at = utc_now()
        self.store.put(job.download_id, job.model_dump(mode="json"))

    def _schedule(self, identity: str) -> None:
        task = self._tasks.get(identity)
        if task is None or task.done():
            self._tasks[identity] = asyncio.create_task(self._run(identity), name=f"public-hub-{identity}")

    async def recover(self) -> None:
        for raw in self.store.list():
            job = ResourceDownloadJob.model_validate(raw)
            if job.state not in TERMINAL_DOWNLOAD_STATES and job.cancel_requested:
                # Cancelled before the restart: honour it instead of silently resuming.
                job.state, job.error = DownloadState.CANCELLED, "Download cancelled"
                self._save(job)
                shutil.rmtree(self.staging / job.download_id, ignore_errors=True)
            elif job.state not in TERMINAL_DOWNLOAD_STATES:
                job.state, job.error, job.cancel_requested = DownloadState.QUEUED, None, False
                self._save(job)
                self._schedule(job.download_id)

    def cancel(self, identity: str) -> ResourceDownloadJob:
        job = self.get(identity)
        if job.state not in TERMINAL_DOWNLOAD_STATES:
            job.cancel_requested = True
            self._cancelled.add(identity)
            self._save(job)
        return job

    def _is_cancelled(self, identity: str) -> bool:
        return identity in self._cancelled or self.get(identity).cancel_requested

    def retry(self, identity: str) -> ResourceDownloadJob:
        job = self.get(identity)
        if job.state not in {DownloadState.ERROR, DownloadState.CANCELLED}:
            raise ValueError("Only failed or cancelled downloads can be retried")
        job.state, job.error, job.cancel_requested = DownloadState.QUEUED, None, False
        job.progress, job.bytes_downloaded, job.retry_count = 0, 0, job.retry_count + 1
        self._cancelled.discard(identity)
        self._save(job)
        self._schedule(identity)
        return job

    async def _run(self, identity: str) -> None:
        async with self._semaphore:
            job = self.get(identity)
            stage = self.staging / identity
            try:
                shutil.rmtree(stage, ignore_errors=True)
                stage.mkdir(parents=True)
                provider = self.providers.get(job.resource.provider)
                job.state = DownloadState.RESOLVING
                self._save(job)
                details = await asyncio.to_thread(provider.details, job.resource)
                job.resolved_revision = details.resolved_revision or job.resource.revision or job.resource.version
                if self._is_cancelled(identity):
                    raise InterruptedError("Download cancelled")
                job.state, job.total_bytes = DownloadState.DOWNLOADING, details.size_bytes
                self._save(job)
                started = time.monotonic()

                def progress(done: int, total: int | None) -> None:
                    current = self.get(identity)
                    current.cancel_requested = current.cancel_requested or identity in self._cancelled
                    current.bytes_downloaded, current.total_bytes = done, total
                    current.progress = min(99, done / total * 100) if total else current.progress
                    elapsed = max(time.monotonic() - started, 0.001)
                    current.speed_bytes_per_second = done / elapsed
                    self._save(current)

                downloaded = await asyncio.to_thread(provider.download, job.resource, stage / "payload", job.requested_format, progress, lambda: self._is_cancelled(identity))
                job = self.get(identity)
                if self._is_cancelled(identity):
                    raise InterruptedError("Download cancelled")
                job.state = DownloadState.VERIFYING
                self._save(job)
                checksum, size = await asyncio.to_thread(tree_identity, downloaded.path)
                if downloaded.provider_checksum and downloaded.path.is_file() and checksum != downloaded.provider_checksum:
                    raise IntegrityError("Downloaded checksum does not match provider checksum")
                if size > self.settings.public_hub_max_extract_bytes:
                    raise IntegrityError("Downloaded resource exceeds configured size limit")
                root = downloaded.path
                archives = [root] if root.is_file() else [path for path in root.iterdir() if path.is_file() and (path.suffix.lower() in {".zip", ".tar", ".tgz", ".gz"})]
                if archives:
                    job.state = DownloadState.EXTRACTING
                    self._save(job)
                    extracted = stage / "extracted"
                    if len(archives) == 1:
                        await asyncio.to_thread(safe_extract_archive, archives[0], extracted, self.settings.public_hub_max_extract_bytes, self.settings.public_hub_max_file_bytes)
                    else:
                        extracted.mkdir()
                        for index, archive in enumerate(archives):
                            await asyncio.to_thread(safe_extract_archive, archive, extracted / str(index), self.settings.public_hub_max_extract_bytes, self.settings.public_hub_max_file_bytes)
                    root = extracted
                    checksum, size = await asyncio.to_thread(tree_identity, root)
                target_parent = self.settings.public_dataset_cache if job.resource.kind == ResourceKind.DATASET else self.settings.public_model_cache
                identity_hash = hashlib.sha256((job.resource.model_dump_json() + (job.requested_format or "") + checksum).encode()).hexdigest()[:24]
                final = target_parent / identity_hash
                temporary = target_parent / f".{identity_hash}.partial"
                self._active_cache_paths.update({final.resolve(), temporary.resolve()})
                try:
                    registered_paths = {Path(raw["local_root"]) for raw in self.assets.list_datasets() + self.assets.list_models() if raw.get("local_root")}
                    await asyncio.to_thread(ensure_cache_capacity, target_parent, size, self.settings.public_hub_max_cache_bytes, registered_paths | self._active_cache_paths)
                    if final.exists():
                        shutil.rmtree(stage, ignore_errors=True)
                    else:
                        shutil.rmtree(temporary, ignore_errors=True)
                        shutil.move(str(root), temporary)
                        temporary.replace(final)
                        shutil.rmtree(stage, ignore_errors=True)
                finally:
                    self._active_cache_paths.discard(final.resolve()); self._active_cache_paths.discard(temporary.resolve())
                job.state = DownloadState.INSPECTING
                self._save(job)
                if job.resource.kind == ResourceKind.DATASET:
                    inspection = await asyncio.to_thread(self.dataset_inspector.inspect, final)
                else:
                    inspection = await asyncio.to_thread(self.model_inspector.inspect, final)
                job.state = DownloadState.REGISTERING
                self._save(job)
                if job.resource.kind == ResourceKind.DATASET:
                    asset = await asyncio.to_thread(self.assets.register_dataset, final, details, inspection)
                else:
                    weight_id = None
                    if self.weights is not None and inspection.state.value == "READY" and inspection.candidate_adapter and inspection.artifact_file:
                        artifact_path = (final / inspection.artifact_file).resolve()
                        if final.resolve() not in artifact_path.parents:
                            raise IntegrityError("Inspected model artifact escapes its cache root")
                        try:
                            with artifact_path.open("rb") as source:
                                weight = await asyncio.to_thread(self.weights.save, source, artifact_path.name, inspection.candidate_adapter)
                            weight_id = weight.weight_id
                        except ValueError:
                            inspection.state = type(inspection.state).ADAPTER_REQUIRED
                            inspection.explanation += "; the detected artifact is not compatible with the existing weight registry"
                    asset = await asyncio.to_thread(self.assets.register_model, final, details, inspection, weight_id)
                job = self.get(identity)
                job.asset_id = getattr(asset, "dataset_id", None) or getattr(asset, "model_artifact_id")
                job.checksum, job.local_path = checksum, str(final)
                job.progress, job.bytes_downloaded, job.total_bytes = 100, size, size
                job.resolved_revision = downloaded.resolved_revision or job.resolved_revision
                job.state = DownloadState.COMPLETED
                self._save(job)
            except InterruptedError:
                job = self.get(identity)
                job.state, job.error = DownloadState.CANCELLED, "Download cancelled"
                self._save(job)
                shutil.rmtree(stage, ignore_errors=True)
            except Exception as exc:
                job = self.get(identity)
                job.state, job.error = DownloadState.ERROR, redact_error(exc)
                self._save(job)
                shutil.rmtree(stage, ignore_errors=True)
