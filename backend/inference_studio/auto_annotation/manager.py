from __future__ import annotations

import asyncio
import hashlib
import os
import shlex
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from inference_studio.public_hub.recipes import RecipeRegistry
from inference_studio.public_hub.schemas import utc_now
from inference_studio.video import VideoReader

from .canonical import canonical_predictions
from .package import PackageBuilder
from .recipe_runner import RecipeExecutor
from .repository import AutoAnnotationRunRepository
from .retention import RetentionEvaluator
from .sampling import FrameSampler
from .schemas import (
    AutoAnnotationItem,
    AutoAnnotationItemState,
    AutoAnnotationRun,
    AutoAnnotationRunCreate,
    AutoAnnotationRunState,
    InferencePreviewRequest,
    SamplingPreviewRequest,
    SourceInspectRequest,
    SourceProvenance,
    SourceType,
)
from .sources import SourceService


_ACTIVE = {
    AutoAnnotationRunState.CREATED,
    AutoAnnotationRunState.VALIDATING,
    AutoAnnotationRunState.PREPARING,
    AutoAnnotationRunState.RUNNING,
    AutoAnnotationRunState.FINALIZING_MEDIA,
    AutoAnnotationRunState.EXPORTING_ANNOTATIONS,
    AutoAnnotationRunState.BUILDING_MANIFEST,
    AutoAnnotationRunState.VALIDATING_PACKAGE,
    AutoAnnotationRunState.CANCELLING,
}


IS_WINDOWS = os.name == "nt"


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def upload_commands(output_root: str, provider: str, destination: str, windows: bool | None = None) -> dict:
    """bash and PowerShell bucket-sync commands for a finished package; ``command`` matches this OS."""
    destination = destination.strip().rstrip("/")
    if any(ord(ch) < 32 for ch in destination):
        raise ValueError("Destination must be a single line without control characters")
    if provider == "gcs":
        if not destination.startswith("gs://"):
            raise ValueError("GCS destination must start with gs://")
        tool = ["gcloud", "storage", "rsync", "-r"]
    elif provider == "s3":
        if not destination.startswith("s3://"):
            raise ValueError("S3 destination must start with s3://")
        tool = ["aws", "s3", "sync"]
    else:
        raise ValueError("Provider must be gcs or s3")
    # The package folder name is portable (AA-<run id>); take it with either separator style.
    name = output_root.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    target = f"{destination}/{name}"
    commands = {
        "bash": " ".join([*tool, shlex.quote(output_root), shlex.quote(target)]),
        # Single quotes: PowerShell performs no $-expansion inside them; '' escapes a quote.
        "powershell": f"{' '.join(tool)} `\n  {_powershell_quote(output_root)} `\n  {_powershell_quote(target)}",
    }
    shell = "powershell" if (IS_WINDOWS if windows is None else windows) else "bash"
    return {"provider": provider, "command": commands[shell], "shell": shell, "commands": commands, "executes_upload": False}


class AutoAnnotationManager:
    def __init__(
        self,
        root: Path,
        output_root: Path,
        sources: SourceService,
        recipes: RecipeRegistry,
        executor: RecipeExecutor,
    ) -> None:
        self.root, self.output_root = root.resolve(), output_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.repository = AutoAnnotationRunRepository(self.root)
        self.sources, self.recipes, self.executor = sources, recipes, executor
        self.packages = PackageBuilder(self.repository)
        self._tasks: dict[str, asyncio.Task] = {}

    def list(self) -> list[dict]:
        return [self.public(run) for run in self.repository.list()]

    def get(self, run_id: str) -> AutoAnnotationRun:
        return self.repository.get(run_id)

    def detail(self, run_id: str) -> dict:
        run = self.get(run_id)
        return {
            **self.public(run),
            "failures": self.repository.failures(run_id),
            "validation": run.package_validation or None,
        }

    def review_items(
        self,
        run_id: str,
        offset: int = 0,
        limit: int = 100,
        prediction_filter: str = "ALL",
    ) -> dict:
        """Return a bounded page of exported items for read-only visual review."""
        self.get(run_id)
        values = [
            item
            for item in self.repository.items_for_run(run_id)
            if item.output_image_relative_path
        ]
        if prediction_filter == "WITH_PREDICTIONS":
            values = [item for item in values if item.prediction_count > 0]
        elif prediction_filter == "WITHOUT_PREDICTIONS":
            values = [item for item in values if item.prediction_count == 0]
        elif prediction_filter != "ALL":
            raise ValueError("Unknown prediction filter")
        page = values[offset : offset + limit]
        return {
            "run_id": run_id,
            "offset": offset,
            "limit": limit,
            "total": len(values),
            "items": [self._review_summary(item) for item in page],
        }

    def review_item(self, run_id: str, item_id: str) -> dict:
        self.get(run_id)
        item = self.repository.get_item(item_id)
        if item.run_id != run_id:
            raise KeyError(item_id)
        value = item.model_dump(mode="json", exclude={"source_path_or_ref"})
        value["image_url"] = self._review_image_url(run_id, item.item_id)
        return value

    def review_image(self, run_id: str, item_id: str) -> Path:
        run = self.get(run_id)
        item = self.repository.get_item(item_id)
        if item.run_id != run_id or not item.output_image_relative_path:
            raise KeyError(item_id)
        root = Path(run.output_root).resolve()
        image = (root / item.output_image_relative_path).resolve()
        if root not in image.parents or not image.is_file():
            raise ValueError("Exported review image is unavailable")
        return image

    @staticmethod
    def _review_image_url(run_id: str, item_id: str) -> str:
        return f"/api/v1/auto-annotation/runs/{run_id}/items/{item_id}/image"

    def _review_summary(self, item: AutoAnnotationItem) -> dict:
        return {
            "item_id": item.item_id,
            "state": item.state.value,
            "source_type": item.source_type.value,
            "source": item.relative_source_path or item.source_video_name or item.item_id,
            "frame_index": item.frame_index,
            "timestamp_seconds": item.timestamp_seconds,
            "width": item.width,
            "height": item.height,
            "prediction_count": item.prediction_count,
            "image_url": self._review_image_url(item.run_id, item.item_id),
        }

    @staticmethod
    def public(run: AutoAnnotationRun) -> dict:
        value = run.model_dump(mode="json")
        total = (
            int(run.source_metadata.get("estimated_sampled_frames", 0))
            if run.source_type == SourceType.VIDEO
            else run.source_count
        )
        total = total or run.sampled_count
        value["progress"] = (
            min(1.0, (run.inferred_count + run.failed_count) / total) if total else 0.0
        )
        value["output_folder"] = run.output_root
        return value

    def create(self, request: AutoAnnotationRunCreate) -> AutoAnnotationRun:
        run_id = f"aa_{uuid.uuid4().hex}"
        output = self.output_root / f"AA-{run_id}"
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        (output / "images").mkdir()
        run = AutoAnnotationRun(
            run_id=run_id,
            source_type=request.source_type,
            source_config=request.source_config,
            recipe_id=request.recipe_id,
            recipe_version=request.recipe_version,
            output_root=str(output),
            sampling_policy=request.sampling_policy,
            retention_policy=request.retention_policy,
            requested_exports=request.requested_exports,
            jpeg_quality=request.jpeg_quality,
        )
        self.repository.create(run)
        self._schedule(run_id)
        return run

    def cancel(self, run_id: str) -> AutoAnnotationRun:
        run = self.get(run_id)
        if run.state not in {
            AutoAnnotationRunState.COMPLETED,
            AutoAnnotationRunState.CANCELLED,
            AutoAnnotationRunState.ERROR,
        }:
            run.cancel_requested, run.state = True, AutoAnnotationRunState.CANCELLING
            self.repository.save(run)
        return run

    def resume(self, run_id: str) -> AutoAnnotationRun:
        run = self.get(run_id)
        if (
            run.state
            not in {AutoAnnotationRunState.ERROR, AutoAnnotationRunState.CANCELLED}
            or not run.recoverable
        ):
            raise ValueError("Only recoverable ERROR or CANCELLED runs can be resumed")
        run.state, run.cancel_requested, run.error, run.completed_at = (
            AutoAnnotationRunState.CREATED,
            False,
            None,
            None,
        )
        self.repository.save(run)
        self._schedule(run_id)
        return run

    async def recover(self) -> None:
        for run in self.repository.list():
            if run.state == AutoAnnotationRunState.CANCELLING:
                run.state, run.completed_at = (
                    AutoAnnotationRunState.CANCELLED,
                    utc_now(),
                )
                self.repository.save(run)
            elif run.state in _ACTIVE:
                run.state = AutoAnnotationRunState.CREATED
                self.repository.save(run)
                self._schedule(run.run_id)

    def _schedule(self, run_id: str) -> None:
        if run_id not in self._tasks or self._tasks[run_id].done():
            self._tasks[run_id] = asyncio.create_task(
                self._run(run_id), name=f"auto-annotation-{run_id}"
            )

    def inspect(self, request: SourceInspectRequest) -> dict:
        return self.sources.inspect(request.source_type, request.source_config)

    def sampling_preview(self, request: SamplingPreviewRequest) -> dict:
        if request.source_type != SourceType.VIDEO:
            raise ValueError("Sampling preview is available only for video")
        return self.sources.preview_sampling(
            request.source_config, request.sampling_policy, request.limit
        )

    async def inference_preview(self, request: InferencePreviewRequest) -> dict:
        preflight = await asyncio.to_thread(
            self.executor.preflight, request.recipe_id, request.recipe_version
        )
        if preflight.status != "READY" or preflight.pin is None:
            return {
                "status": "NOT_READY",
                "preflight": preflight.model_dump(mode="json"),
                "items": [],
            }
        recipe = self.recipes.get(request.recipe_id, request.recipe_version)
        if request.source_type == SourceType.IMAGE_FOLDER:
            samples = await asyncio.to_thread(
                self.sources.preview_images, request.source_config, request.limit
            )
        else:
            if request.sampling_policy is None:
                raise ValueError("Video inference preview requires a sampling policy")
            samples = await asyncio.to_thread(
                self.sources.preview_video_frames,
                request.source_config,
                request.sampling_policy,
                request.limit,
            )
        evaluator, values = RetentionEvaluator(request.retention_policy), []
        for index, (metadata, image) in enumerate(samples):
            source_id = str(metadata.get("item_id") or f"preview-{index}")
            source = self._provenance(request.source_type, metadata)
            started = perf_counter()
            result = await self.executor.execute(
                recipe,
                image,
                str(
                    metadata.get("relative_path")
                    or metadata.get("source_video_name")
                    or "preview.jpg"
                ),
                source_id,
                persist_preview=True,
            )
            predictions = canonical_predictions(
                result["result"], f"preview_{index}", preflight.pin, source
            )
            timestamp = metadata.get("timestamp_seconds")
            decision = evaluator.evaluate(predictions, timestamp)
            values.append(
                {
                    "source": metadata,
                    "preview_url": result.get("preview_url"),
                    "prediction_count": len(predictions),
                    "classes": sorted({value.category_name for value in predictions}),
                    "confidences": [value.confidence for value in predictions],
                    "retained": decision.retained,
                    "retention_reason": decision.reason,
                    "latency_ms": (perf_counter() - started) * 1000,
                }
            )
        return {
            "status": "READY",
            "preflight": preflight.model_dump(mode="json"),
            "items": values,
        }

    async def _run(self, run_id: str) -> None:
        run = self.get(run_id)
        try:
            run.state, run.started_at = (
                AutoAnnotationRunState.VALIDATING,
                run.started_at or utc_now(),
            )
            self.repository.save(run)
            preflight = await asyncio.to_thread(
                self.executor.preflight, run.recipe_id, run.recipe_version
            )
            if preflight.status != "READY" or preflight.pin is None:
                raise RuntimeError(
                    "Recipe preflight failed: "
                    + "; ".join(issue.message for issue in preflight.errors)
                )
            run.recipe_pin, run.recipe_hash = preflight.pin, preflight.pin.recipe_hash
            recipe = self.recipes.get(run.recipe_id, run.recipe_version)
            run.state = AutoAnnotationRunState.PREPARING
            inspected_source = await asyncio.to_thread(
                self.sources.inspect, run.source_type, run.source_config
            )
            previous_snapshot = run.source_metadata.get("snapshot_sha256")
            if (
                previous_snapshot
                and inspected_source.get("snapshot_sha256") != previous_snapshot
            ):
                raise RuntimeError(
                    "Source snapshot changed since this run was created; start a new run"
                )
            run.source_metadata = inspected_source
            run.source_count = int(
                run.source_metadata.get("source_count")
                or run.source_metadata.get("total_frames")
                or 0
            )
            if run.source_type == SourceType.VIDEO and run.sampling_policy is not None:
                run.source_metadata["estimated_sampled_frames"] = (
                    FrameSampler.estimated_count(
                        run.sampling_policy,
                        float(run.source_metadata["nominal_fps"]),
                        int(run.source_metadata["total_frames"]),
                        float(run.source_metadata["duration_seconds"]),
                    )
                )
            self.repository.save(run)
            evaluator = RetentionEvaluator(run.retention_policy)
            prior = [
                item.timestamp_seconds
                for item in self.repository.items_for_run(run_id)
                if item.state == AutoAnnotationItemState.EXPORTED
                and item.timestamp_seconds is not None
            ]
            evaluator.last_retained_timestamp = max(prior) if prior else None
            run.state = AutoAnnotationRunState.RUNNING
            self.repository.save(run)
            if run.source_type == SourceType.IMAGE_FOLDER:
                await self._run_images(run, recipe, evaluator)
            else:
                await self._run_video(run, recipe, evaluator)
            run = self.get(run_id)
            if run.cancel_requested:
                run.state, run.completed_at = (
                    AutoAnnotationRunState.CANCELLED,
                    utc_now(),
                )
                self._refresh_counts(run)
                self.repository.save(run)
                return
            run.state = AutoAnnotationRunState.FINALIZING_MEDIA
            self.repository.save(run)
            run.state = AutoAnnotationRunState.EXPORTING_ANNOTATIONS
            self.repository.save(run)
            exports = await asyncio.to_thread(self.packages.export_annotations, run)
            run.state = AutoAnnotationRunState.BUILDING_MANIFEST
            self.repository.save(run)
            manifest_sha = await asyncio.to_thread(self.packages.build_manifest, run)
            run.state = AutoAnnotationRunState.VALIDATING_PACKAGE
            self.repository.save(run)
            self._refresh_counts(run)
            run.state, run.ready_for_upload, run.completed_at = (
                AutoAnnotationRunState.COMPLETED,
                True,
                utc_now(),
            )
            await asyncio.to_thread(
                self.packages.write_metadata, run, exports, manifest_sha
            )
            validation = await asyncio.to_thread(self.packages.validate, run)
            run.package_validation = validation
            if not validation["valid"]:
                self.repository.save(run)
                raise RuntimeError(
                    "Package validation failed: " + "; ".join(validation["errors"])
                )
            self.repository.save(run)
        except Exception as exc:
            run = self.get(run_id)
            self._refresh_counts(run)
            run.state, run.error, run.completed_at = (
                AutoAnnotationRunState.ERROR,
                str(exc),
                utc_now(),
            )
            self.repository.save(run)

    async def _run_images(
        self, run: AutoAnnotationRun, recipe, evaluator: RetentionEvaluator
    ) -> None:
        _, inventory = await asyncio.to_thread(
            self.sources.image_inventory, run.source_config
        )
        if self.sources.inventory_sha256(inventory) != run.source_metadata.get(
            "snapshot_sha256"
        ):
            raise RuntimeError("Image-folder source changed while preparing the run")
        consecutive_failures = 0
        for metadata in inventory:
            if self.get(run.run_id).cancel_requested:
                return
            item_id = f"aai_{run.run_id[3:15]}_{metadata['item_id']}"
            item = self._existing_or_new(
                item_id, run, metadata, SourceType.IMAGE_FOLDER
            )
            if item.state in {
                AutoAnnotationItemState.EXPORTED,
                AutoAnnotationItemState.DISCARDED,
                AutoAnnotationItemState.FAILED,
            }:
                continue
            image = (
                await asyncio.to_thread(cv2.imread, metadata["path"])
                if not metadata.get("error")
                else None
            )
            if image is None:
                item.state, item.error = (
                    AutoAnnotationItemState.FAILED,
                    metadata.get("error", "Image cannot be decoded"),
                )
                item.error_stage = "SOURCE"
                self.repository.record_item(item)
                self._refresh_save(run, item)
                continue
            succeeded = await self._process(run, recipe, evaluator, item, image)
            consecutive_failures = 0 if succeeded else consecutive_failures + 1
            if consecutive_failures >= 3:
                raise RuntimeError(
                    "Inference failed for three consecutive source items; stopping as a systemic execution failure"
                )

    async def _run_video(
        self, run: AutoAnnotationRun, recipe, evaluator: RetentionEvaluator
    ) -> None:
        path, record = self.sources.video_path(run.source_config)
        actual_sha = await asyncio.to_thread(self.sources.file_sha256, path)
        if actual_sha != run.source_metadata.get("snapshot_sha256"):
            raise RuntimeError("Video source changed while preparing the run")
        reader = VideoReader(path)
        await asyncio.to_thread(reader.open)
        try:
            sampler = FrameSampler(run.sampling_policy, reader.fps)  # type: ignore[arg-type]
            consecutive_failures = 0
            frames = reader.frames()
            while True:
                packet = await asyncio.to_thread(next, frames, None)
                if packet is None:
                    break
                if self.get(run.run_id).cancel_requested:
                    return
                if not sampler.select(packet.index, packet.timestamp_seconds):
                    continue
                item_id = f"aai_{run.run_id[3:15]}_{packet.index:012d}"
                metadata = {
                    "path": str(path),
                    "sha256": actual_sha,
                    "frame_index": packet.index,
                    "pts": packet.pts,
                    "time_base": packet.time_base,
                    "timestamp_seconds": packet.timestamp_seconds,
                    "width": packet.image.shape[1],
                    "height": packet.image.shape[0],
                    "source_video_id": record.asset_id
                    if record
                    else hashlib.sha256(str(path).encode()).hexdigest()[:24],
                    "source_video_name": record.filename if record else path.name,
                }
                item = self._existing_or_new(item_id, run, metadata, SourceType.VIDEO)
                if item.state in {
                    AutoAnnotationItemState.EXPORTED,
                    AutoAnnotationItemState.DISCARDED,
                    AutoAnnotationItemState.FAILED,
                }:
                    continue
                succeeded = await self._process(
                    run, recipe, evaluator, item, packet.image
                )
                consecutive_failures = 0 if succeeded else consecutive_failures + 1
                if consecutive_failures >= 3:
                    raise RuntimeError(
                        "Inference failed for three consecutive sampled frames; stopping as a systemic execution failure"
                    )
        finally:
            await asyncio.to_thread(reader.close)

    def _existing_or_new(
        self,
        item_id: str,
        run: AutoAnnotationRun,
        metadata: dict,
        source_type: SourceType,
    ) -> AutoAnnotationItem:
        try:
            return self.repository.get_item(item_id)
        except KeyError:
            pass
        item = AutoAnnotationItem(
            item_id=item_id,
            run_id=run.run_id,
            source_type=source_type,
            source_path_or_ref=str(
                run.source_config.asset_id or metadata.get("path", "configured source")
            ),
            source_sha256=metadata.get("sha256"),
            relative_source_path=metadata.get("relative_path"),
            source_video_id=metadata.get("source_video_id"),
            source_video_name=metadata.get("source_video_name"),
            frame_index=metadata.get("frame_index"),
            pts=metadata.get("pts"),
            time_base=metadata.get("time_base"),
            timestamp_seconds=metadata.get("timestamp_seconds"),
            width=int(metadata.get("width", 0)),
            height=int(metadata.get("height", 0)),
            state=AutoAnnotationItemState.SAMPLED,
        )
        self.repository.record_item(item)
        latest = self.get(run.run_id)
        latest.sampled_count += 1
        self.repository.save(latest)
        return item

    async def _process(
        self,
        run: AutoAnnotationRun,
        recipe,
        evaluator: RetentionEvaluator,
        item: AutoAnnotationItem,
        image: Any,
    ) -> bool:
        item.state, item.error, item.error_stage = (
            AutoAnnotationItemState.PROCESSING,
            None,
            None,
        )
        self.repository.record_item(item)
        started = perf_counter()
        stage = "INFERENCE"
        try:
            result = await self.executor.execute(
                recipe,
                image,
                item.relative_source_path or item.source_video_name or item.item_id,
                item.item_id,
                persist_preview=False,
            )
            stage = "CANONICALIZATION"
            source = self._provenance(item.source_type, item.model_dump(mode="json"))
            item.predictions = canonical_predictions(
                result["result"], item.item_id, run.recipe_pin, source
            )  # type: ignore[arg-type]
            item.prediction_count, item.state = (
                len(item.predictions),
                AutoAnnotationItemState.INFERRED,
            )
            item.metrics = {
                **result.get("metrics", {}),
                "total_ms": (perf_counter() - started) * 1000,
            }
            self.repository.record_item(item)
            stage = "RETENTION"
            decision = evaluator.evaluate(item.predictions, item.timestamp_seconds)
            if not decision.retained:
                item.state, item.discarded_reason = (
                    AutoAnnotationItemState.DISCARDED,
                    decision.reason,
                )
            else:
                item.state, item.retained_reason = (
                    AutoAnnotationItemState.RETAINED,
                    decision.reason,
                )
                self.repository.record_item(item)
                stage = "MEDIA_EXPORT"
                item.output_image_relative_path = await asyncio.to_thread(
                    self._write_image, run, item, image
                )
                item.state = AutoAnnotationItemState.EXPORTED
            self.repository.record_item(item)
        except Exception as exc:
            item.state, item.error = AutoAnnotationItemState.FAILED, str(exc)
            item.error_stage = stage
            self.repository.record_item(item)
            self._refresh_save(run, item)
            if self._systemic_error(str(exc)):
                raise RuntimeError(f"Systemic inference failure: {exc}") from exc
            return False
        self._refresh_save(run, item)
        return True

    @staticmethod
    def _systemic_error(message: str) -> bool:
        lowered = message.casefold()
        signals = (
            "checkpoint",
            "weight",
            "model load",
            "adapter",
            "cuda",
            "out of memory",
            "no module named",
            "dependency",
        )
        return any(signal in lowered for signal in signals)

    @staticmethod
    def _provenance(source_type: SourceType, value: dict) -> SourceProvenance:
        return SourceProvenance(
            source_type=source_type,
            source_image=value.get("relative_path")
            or value.get("relative_source_path"),
            source_sha256=value.get("sha256") or value.get("source_sha256"),
            source_video_id=value.get("source_video_id"),
            source_video_name=value.get("source_video_name"),
            source_frame_index=value.get("frame_index"),
            source_timestamp_seconds=value.get("timestamp_seconds"),
            source_pts=value.get("pts"),
            source_time_base=value.get("time_base"),
        )

    @staticmethod
    def _write_image(
        run: AutoAnnotationRun, item: AutoAnnotationItem, image: Any
    ) -> str:
        name = f"{item.item_id}.jpg"
        relative = Path("images") / name
        target = Path(run.output_root) / relative
        temporary = target.with_suffix(".tmp")
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, run.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode retained image")
        with temporary.open("wb") as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return relative.as_posix()

    def _refresh_save(
        self, run: AutoAnnotationRun, item: AutoAnnotationItem | None = None
    ) -> None:
        latest = self.get(run.run_id)
        if item:
            if item.state == AutoAnnotationItemState.EXPORTED:
                latest.inferred_count += 1
                latest.retained_count += 1
                latest.exported_count += 1
                latest.prediction_count += item.prediction_count
            elif item.state == AutoAnnotationItemState.DISCARDED:
                latest.inferred_count += 1
                latest.prediction_count += item.prediction_count
            elif item.state == AutoAnnotationItemState.FAILED:
                latest.failed_count += 1
            latest.last_item = {
                "item_id": item.item_id,
                "state": item.state.value,
                "source": item.relative_source_path or item.source_video_name,
                "frame_index": item.frame_index,
            }
        self.repository.save(latest)

    def _refresh_counts(self, run: AutoAnnotationRun) -> None:
        items = self.repository.items_for_run(run.run_id)
        run.sampled_count = len(items)
        run.inferred_count = sum(
            item.state
            in {
                AutoAnnotationItemState.INFERRED,
                AutoAnnotationItemState.RETAINED,
                AutoAnnotationItemState.EXPORTED,
                AutoAnnotationItemState.DISCARDED,
            }
            for item in items
        )
        run.retained_count = sum(
            item.state
            in {AutoAnnotationItemState.RETAINED, AutoAnnotationItemState.EXPORTED}
            for item in items
        )
        run.exported_count = sum(
            item.state == AutoAnnotationItemState.EXPORTED for item in items
        )
        run.failed_count = sum(
            item.state == AutoAnnotationItemState.FAILED for item in items
        )
        run.prediction_count = sum(
            item.prediction_count
            for item in items
            if item.state != AutoAnnotationItemState.FAILED
        )

    def upload_command(self, run_id: str, provider: str, destination: str) -> dict:
        """Display-only bucket sync commands (bash and PowerShell); nothing is executed by the studio."""
        run = self.get(run_id)
        if not run.ready_for_upload:
            raise ValueError("Run package is not ready for upload")
        return upload_commands(run.output_root, provider, destination)
