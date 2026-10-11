from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2

from inference_studio.adapters import ModelConfig
from inference_studio.assets import AssetStore
from inference_studio.chainner import ChainnerImageAdapter, ChainnerNodeExecutor, ChainnerOutputs
from inference_studio.models import ModelManager
from inference_studio.nodes.runtime import ImageValue, ModelOutput, VideoModelStream, VideoValue
from inference_studio.renderers import RendererRegistry
from inference_studio.results import FeatureResult, VideoFrameResult
from inference_studio.runtime import ExecutionScope, FeatureArtifact, FeatureStore, FeatureValue, HeatmapValue
from inference_studio.features.ops import (
    activation_xor, difference,
    mahalanobis_difference, normalize,
)
from inference_studio.features.visualization import feature_visual as feature_heatmap, render_heatmap
from inference_studio.video import VideoMetricsCollector, VideoReader

from .schema import Workflow
from .planner import ExecutionPlan, build_execution_plan
from .validator import WorkflowValidator



def _synced_now(artifact: Any) -> float:
    """perf_counter() after the artifact's CUDA stream drains, so feature-op timings are real, not queue time."""
    if str(getattr(artifact.metadata, "device", "")).startswith("cuda"):
        import torch
        torch.cuda.synchronize(torch.device(artifact.metadata.device))
    return perf_counter()

class RunState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass(eq=False)
class ExecutionSubscriber:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    frame_ready: asyncio.Event = field(default_factory=asyncio.Event)
    latest_frame: dict[str, Any] | None = None


@dataclass
class Execution:
    run_id: str
    workflow: Workflow
    state: RunState = RunState.QUEUED
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[ExecutionSubscriber] = field(default_factory=list)
    pause_gate: asyncio.Event = field(default_factory=asyncio.Event)
    stop_requested: bool = False
    pause_requested: bool = False
    scope: ExecutionScope | None = None
    plan: ExecutionPlan | None = None

    def __post_init__(self) -> None:
        self.pause_gate.set()


class ExecutionEngine:
    def __init__(self, validator: WorkflowValidator, assets: AssetStore, manager: ModelManager, preview_dir: Path, benchmark_dir: Path | None = None, preview_retention: int = 10, chainner: ChainnerNodeExecutor | None = None, settings_snapshot: Callable[[], dict[str, Any]] | None = None, feature_store_max_entries: int = 64, feature_store_max_bytes: int = 2 * 1024**3, geometries: Any = None) -> None:
        self.validator, self.assets, self.manager = validator, assets, manager
        self.preview_dir = preview_dir
        self.benchmark_dir = benchmark_dir or preview_dir.parent / "benchmarks"
        self.preview_retention = max(1, preview_retention)
        self.renderers = RendererRegistry()
        self.feature_store = FeatureStore(feature_store_max_entries, feature_store_max_bytes)
        self.geometries = geometries
        self.chainner = chainner
        self.settings_snapshot = settings_snapshot or (lambda: {})
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.executions: dict[str, Execution] = {}

    def start(self, workflow: Workflow) -> Execution:
        validation = self.validator.validate(workflow)
        if not validation.valid:
            raise ValueError("; ".join(issue.message for issue in validation.issues))
        run_id = f"run_{uuid.uuid4().hex}"
        plan = build_execution_plan(workflow, self.validator.nodes)
        execution = Execution(run_id=run_id, workflow=workflow, scope=ExecutionScope(run_id, self.feature_store, self.settings_snapshot()), plan=plan)
        self.executions[execution.run_id] = execution
        asyncio.create_task(self._execute(execution, plan.order))
        return execution

    async def publish(self, execution: Execution, event: dict[str, Any]) -> None:
        event = {"run_id": execution.run_id, **event}
        if event.get("type") == "video_frame":
            # Retain and deliver only the newest live frame. Control/lifecycle
            # events remain lossless and use a separate queue.
            execution.events = [item for item in execution.events if item.get("type") != "video_frame"][-499:]
            execution.events.append(event)
            for subscriber in list(execution.subscribers):
                subscriber.latest_frame = event
                subscriber.frame_ready.set()
            return
        execution.events.append(event)
        execution.events = execution.events[-500:]
        for subscriber in list(execution.subscribers):
            await subscriber.queue.put(event)

    async def subscribe(self, run_id: str):
        execution = self.get(run_id)
        subscriber = ExecutionSubscriber()
        execution.subscribers.append(subscriber)
        terminal = {RunState.COMPLETED, RunState.ERROR, RunState.STOPPED, RunState.CANCELLED}
        try:
            # Snapshot after registering the queue: later events cannot fall into
            # the gap between replay and live delivery.
            for event in list(execution.events):
                yield event
                if event.get("type") == "run_state" and event.get("state") in terminal:
                    return
            while True:
                if subscriber.latest_frame is not None:
                    subscriber.frame_ready.clear()
                    event, subscriber.latest_frame = subscriber.latest_frame, None
                else:
                    control_task = asyncio.create_task(subscriber.queue.get())
                    frame_task = asyncio.create_task(subscriber.frame_ready.wait())
                    done, pending = await asyncio.wait({control_task, frame_task}, return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if control_task in done:
                        event = control_task.result()
                    else:
                        subscriber.frame_ready.clear()
                        event, subscriber.latest_frame = subscriber.latest_frame, None
                        if event is None:
                            continue
                yield event
                if event.get("type") == "run_state" and event.get("state") in terminal:
                    return
        finally:
            execution.subscribers.remove(subscriber)

    def get(self, run_id: str) -> Execution:
        try:
            return self.executions[run_id]
        except KeyError as exc:
            raise ValueError(f"Unknown run: {run_id}") from exc

    async def pause(self, run_id: str) -> None:
        execution = self.get(run_id)
        if execution.state != RunState.RUNNING:
            return
        execution.pause_requested = True
        execution.pause_gate.clear()
        await self.publish(execution, {"type": "execution_control", "event": "pause_requested", "state": execution.state})

    async def resume(self, run_id: str) -> None:
        execution = self.get(run_id)
        if execution.state != RunState.PAUSED and not execution.pause_requested:
            return
        execution.pause_requested = False
        execution.state = RunState.RUNNING
        execution.pause_gate.set()
        await self.publish(execution, {"type": "run_state", "event": "workflow_resumed", "state": execution.state})

    async def stop(self, run_id: str) -> None:
        execution = self.get(run_id)
        if execution.state in {RunState.COMPLETED, RunState.ERROR, RunState.STOPPED, RunState.CANCELLED}:
            return
        execution.stop_requested = True
        execution.pause_gate.set()
        await self.publish(execution, {"type": "execution_control", "event": "cancel_requested", "state": execution.state})

    async def _checkpoint(self, execution: Execution) -> None:
        if execution.stop_requested:
            raise InterruptedError("Execution stopped")
        if execution.pause_requested:
            execution.state = RunState.PAUSED
            await self.publish(execution, {"type": "run_state", "event": "workflow_paused", "state": execution.state})
        await execution.pause_gate.wait()
        if execution.stop_requested:
            raise InterruptedError("Execution stopped")

    async def _execute(self, execution: Execution, order: list[str]) -> None:
        execution.state = RunState.RUNNING
        await self.publish(execution, {"type": "run_state", "event": "workflow_started", "state": execution.state})
        values: dict[str, Any] = {}
        node_map = {node.id: node for node in execution.workflow.nodes}
        for node_id in order:
            await self.publish(execution, {"type": "node_state", "event": "node_waiting", "node_id": node_id, "state": "waiting"})
        current_node_id: str | None = None
        try:
            for node_id in order:
                await self._checkpoint(execution)
                current_node_id = node_id
                node = node_map[node_id]
                node_started = perf_counter()
                state = "loading" if node.type == "model" else "running"
                await self.publish(execution, {"type": "node_state", "event": "node_started", "node_id": node_id, "state": state})
                incoming = {edge.target_handle: self._edge_value(values[edge.source], edge.source_handle) for edge in execution.workflow.edges if edge.target == node_id}
                try:
                    values[node_id] = await self._run_node(execution, node, incoming)
                finally:
                    for value in incoming.values():
                        if isinstance(value, FeatureValue):
                            self.feature_store.release(value.feature_id)
                    if execution.scope is not None:
                        await execution.scope.close_node(node_id)
                await self.publish(execution, {"type": "node_state", "event": "node_completed", "node_id": node_id, "state": "success", "duration_ms": (perf_counter() - node_started) * 1000})
            execution.state = RunState.COMPLETED
            await self.publish(execution, {"type": "run_state", "event": "workflow_completed", "state": execution.state})
        except InterruptedError:
            execution.state = RunState.CANCELLED
            await self.publish(execution, {"type": "run_state", "event": "workflow_cancelled", "state": execution.state})
        except Exception as exc:
            execution.state = RunState.ERROR
            if current_node_id is not None:
                await self.publish(execution, {"type": "node_state", "event": "node_failed", "node_id": current_node_id, "state": "error", "error": str(exc)})
            await self.publish(execution, {"type": "run_state", "event": "workflow_failed", "state": execution.state, "error": str(exc)})
        finally:
            if execution.scope is not None:
                try:
                    await execution.scope.close()
                except Exception as exc:
                    await self.publish(execution, {"type": "cleanup_error", "event": "cleanup_failed", "error": str(exc)})

    @staticmethod
    def _edge_value(value: Any, source_handle: str) -> Any:
        if isinstance(value, ChainnerOutputs):
            try:
                return value.values[source_handle]
            except KeyError as exc:
                raise ValueError(f"Unknown chaiNNer output port: {source_handle}") from exc
        return value

    async def _run_node(self, execution: Execution, node, inputs: dict[str, Any]):
        if node.type in {"image_source", "video_source"}:
            asset = self.assets.get(str(node.config["asset_id"]))
            if node.type == "video_source":
                return VideoValue(asset=asset, frame_stride=int(node.config.get("frame_stride", 1)), warmup_frames=int(node.config.get("warmup_frames", 2)), benchmark_frames=int(node.config.get("benchmark_frames", 0)))
            image = await asyncio.to_thread(cv2.imread, asset.path)
            if image is None:
                raise RuntimeError("Image asset can no longer be decoded")
            return ImageValue(image=image, asset=asset)
        if node.type == "resize":
            value = inputs["image"]
            image = await asyncio.to_thread(cv2.resize, value.image, (int(node.config["width"]), int(node.config["height"])))
            return ImageValue(image=image, asset=value.asset)
        if node.type == "crop":
            value = inputs["image"]
            x, y = int(node.config.get("x", 0)), int(node.config.get("y", 0))
            width, height = int(node.config["width"]), int(node.config["height"])
            image = value.image[y:y + height, x:x + width]
            if image.size == 0:
                raise ValueError("Crop is outside the image")
            return ImageValue(image=image, asset=value.asset)
        if node.type == "threshold":
            value = inputs["image"]
            gray = cv2.cvtColor(value.image, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, float(node.config.get("value", 127)), 255, cv2.THRESH_BINARY)
            return ImageValue(image=cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), asset=value.asset)
        if node.type.startswith("chainner:"):
            if self.chainner is None:
                raise RuntimeError("chaiNNer integration is not configured")
            return await self.chainner.execute(node.type, node.config, inputs, execution, node.id, self.publish)
        if node.type == "feature_extract":
            value: ImageValue = inputs["image"]
            provider = str(node.config["provider"])
            weight_id = str(node.config["weight_id"])
            device = str(node.config.get("device", "cpu"))
            precision = str(node.config.get("precision", "fp32"))
            stage = str(node.config["stage"])
            reused = self.manager.feature_load_count(provider, weight_id, device, precision) > 0
            artifact = await self.manager.extract_feature(
                provider, weight_id, ChainnerImageAdapter.to_vision(value.image),
                stage, device, precision,
            )
            count = execution.plan.consumer_count(node.id, "feature_map") if execution.plan else 1
            store_started = perf_counter()
            feature_value = self.feature_store.put(artifact, execution.run_id, count)
            store_ms = (perf_counter() - store_started) * 1000
            public = FeatureResult(
                shape=list(artifact.metadata.shape), feature_id=feature_value.feature_id,
                dtype=artifact.metadata.dtype, device=artifact.metadata.device,
                memory_bytes=artifact.metadata.memory_bytes, metadata=artifact.metadata.public(),
                summary=artifact.summary, metrics=artifact.metrics,
            )
            await self.publish(execution, {
                "type": "inference_result", "event": "feature_extracted", "node_id": node.id,
                "provider": provider, "stage": stage, "model_reused": reused,
                "feature_store_ms": store_ms, "result": public.model_dump(mode="json"),
            })
            return feature_value
        if node.type == "feature_normalize":
            value: FeatureValue = inputs["feature"]
            started = perf_counter()
            artifact = normalize(self.feature_store.get(value.feature_id), float(node.config.get("epsilon", 1e-8)), str(node.config.get("mode", "l2_channel")))
            compute_ms = (perf_counter() - started) * 1000
            count = execution.plan.consumer_count(node.id, "feature") if execution.plan else 1
            feature_value = self.feature_store.put(artifact, execution.run_id, count)
            await self.publish(execution, {"type": "feature_metrics", "event": "feature_normalized", "node_id": node.id, "compute_ms": compute_ms, "memory_bytes": artifact.metadata.memory_bytes})
            return feature_value
        if node.type == "feature_difference":
            reference, target = self.feature_store.get(inputs["reference"].feature_id), self.feature_store.get(inputs["current"].feature_id)
            started = _synced_now(reference)
            artifact, heatmap = difference(reference, target, str(node.config.get("mode", "l2")))
            compute_ms = (_synced_now(reference) - started) * 1000
            count = execution.plan.consumer_count(node.id, "delta") if execution.plan else 1
            feature_value = self.feature_store.put(artifact, execution.run_id, count)
            await self.publish(execution, {"type": "feature_metrics", "event": "feature_difference_computed", "node_id": node.id, "compute_ms": compute_ms, "memory_bytes": artifact.metadata.memory_bytes, "statistics": heatmap.statistics})
            return ChainnerOutputs({"delta": feature_value, "heatmap": heatmap})
        if node.type == "rfdetr_activation_xor":
            reference = self.feature_store.get(inputs["reference"].feature_id)
            current = self.feature_store.get(inputs["current"].feature_id)
            started = _synced_now(reference)
            heatmap = activation_xor(reference, current, float(node.config.get("percentile", 90)))
            await self.publish(execution, {
                "type": "feature_metrics", "event": "activation_xor_computed",
                "node_id": node.id, "xor_ms": (_synced_now(reference) - started) * 1000,
                "statistics": heatmap.statistics,
            })
            return heatmap
        if node.type == "mahalanobis_difference":
            reference = self.feature_store.get(inputs["reference"].feature_id)
            current = self.feature_store.get(inputs["current"].feature_id)
            if self.geometries is None:
                raise RuntimeError("Mahalanobis geometry store is not configured")
            geometry = await asyncio.to_thread(self.geometries.get, str(node.config["geometry_id"]))
            started = _synced_now(reference)
            heatmap = mahalanobis_difference(reference, current, geometry)
            await self.publish(execution, {
                "type": "feature_metrics", "event": "mahalanobis_computed",
                "node_id": node.id, "mahalanobis_ms": (_synced_now(reference) - started) * 1000,
                "geometry_id": str(node.config["geometry_id"]),
                "statistics": heatmap.statistics,
            })
            return heatmap
        if node.type == "model":
            config = ModelConfig.model_validate(node.config)
            value = inputs["input"]
            if isinstance(value, VideoValue):
                reused = self.manager.state_for(config).value == "LOADED"
                await self.manager.load(config)
                await self.publish(execution, {"type": "model_lifecycle", "event": "model_reused" if reused else "model_loaded", "node_id": node.id, "model": config.model_dump(mode="json")})
                return VideoModelStream(asset=value.asset, config=config, frame_stride=value.frame_stride, warmup_frames=value.warmup_frames, benchmark_frames=value.benchmark_frames)
            await self.manager.load(config)
            await self.publish(execution, {"type": "node_state", "event": "node_started", "node_id": node.id, "state": "running"})
            image = ChainnerImageAdapter.to_vision(value.image)
            result = await self.manager.predict(config, image)
            if isinstance(result, FeatureArtifact):
                count = execution.plan.consumer_count(node.id, "output") if execution.plan else 1
                feature_value = self.feature_store.put(result, execution.run_id, count)
                public = FeatureResult(shape=list(result.metadata.shape), feature_id=feature_value.feature_id, dtype=result.metadata.dtype, device=result.metadata.device, memory_bytes=result.metadata.memory_bytes, metadata=result.metadata.public(), summary=result.summary, metrics=result.metrics)
                await self.publish(execution, {"type": "inference_result", "event": "inference_completed", "node_id": node.id, "model": config.model_dump(mode="json"), "result": public.model_dump(mode="json")})
                return feature_value
            await self.publish(execution, {"type": "inference_result", "event": "inference_completed", "node_id": node.id, "model": config.model_dump(mode="json"), "result": result.model_dump(mode="json")})
            return ModelOutput(image=image, result=result, config=config)
        if node.type == "video_preview":
            return await self._run_video(execution, node.id, inputs["result"])
        if node.type == "image_preview":
            image = ChainnerImageAdapter.to_vision(inputs["image"].image)
            original_url = await self._save_preview(execution.run_id, node.id, image, label="original")
            prediction_url = await self._save_preview(execution.run_id, node.id, image, label="prediction")
            await self.publish(execution, {"type": "preview", "event": "preview_ready", "node_id": node.id, "url": prediction_url, "original_url": original_url, "prediction_url": prediction_url})
            return {"original_url": original_url, "prediction_url": prediction_url}
        if node.type == "feature_preview":
            value: FeatureValue = inputs["result"]
            artifact = self.feature_store.get(value.feature_id)
            started = perf_counter()
            values = feature_heatmap(artifact, str(node.config.get("mode", "l2")), int(node.config.get("channel", 0)))
            canvas = render_heatmap(values, artifact.metadata.source_width, artifact.metadata.source_height)
            prediction_url = await self._save_preview(execution.run_id, node.id, canvas, label="prediction")
            result = FeatureResult(shape=list(artifact.metadata.shape), feature_id=value.feature_id, dtype=artifact.metadata.dtype, device=artifact.metadata.device, memory_bytes=artifact.metadata.memory_bytes, metadata=artifact.metadata.public(), summary=artifact.summary)
            await self.publish(execution, {"type": "preview", "event": "preview_ready", "node_id": node.id, "url": prediction_url, "prediction_url": prediction_url, "result": result.model_dump(mode="json"), "render_ms": (perf_counter() - started) * 1000})
            return {"prediction_url": prediction_url}
        if node.type == "heatmap_preview":
            value: HeatmapValue = inputs["heatmap"]
            started = perf_counter()
            canvas = render_heatmap(value.values, value.source_width, value.source_height)
            prediction_url = await self._save_preview(execution.run_id, node.id, canvas, label="prediction")
            await self.publish(execution, {"type": "preview", "event": "preview_ready", "node_id": node.id, "url": prediction_url, "prediction_url": prediction_url, "result": {"result_type": "heatmap", "statistics": value.statistics, "shape": [value.grid_height, value.grid_width]}, "render_ms": (perf_counter() - started) * 1000})
            return {"prediction_url": prediction_url}
        output = inputs["result"]
        expected = {"detection_preview": "detections", "segmentation_preview": "segmentation", "feature_preview": ("feature_map", "embedding"), "tensor_preview": "tensor"}[node.type]
        if output.result.result_type not in ((expected,) if isinstance(expected, str) else expected):
            raise ValueError(f"{node.type} cannot render {output.result.result_type}")
        canvas = self.renderers.render(output.image, output.result)
        original_url = await self._save_preview(execution.run_id, node.id, output.image, label="original")
        prediction_url = await self._save_preview(execution.run_id, node.id, canvas, label="prediction")
        await self.publish(execution, {"type": "preview", "event": "preview_ready", "node_id": node.id, "url": prediction_url, "original_url": original_url, "prediction_url": prediction_url, "model": output.config.model_dump(mode="json"), "result": output.result.model_dump(mode="json")})
        return {"original_url": original_url, "prediction_url": prediction_url}

    async def _run_video(self, execution: Execution, node_id: str, stream: VideoModelStream):
        reader = VideoReader(stream.asset.path)
        source = {"asset_id": stream.asset.asset_id, "filename": stream.asset.filename, "codec": stream.asset.codec, "width": stream.asset.width, "height": stream.asset.height, "fps": stream.asset.fps, "frame_count": stream.asset.frame_count, "duration_seconds": stream.asset.duration_seconds, "frame_stride": stream.frame_stride}
        model = stream.config.model_dump(mode="json")
        collector = VideoMetricsCollector(source, model, stream.warmup_frames)
        report: dict[str, Any] | None = None
        final_state = "completed"
        try:
            await asyncio.to_thread(reader.open)
            await self.publish(execution, {"type": "video_lifecycle", "event": "video_opened", "node_id": node_id, "source": source})
            await self.publish(execution, {"type": "video_lifecycle", "event": "video_started", "node_id": node_id, "frame_stride": stream.frame_stride, "warmup_frames": stream.warmup_frames, "benchmark_frames": stream.benchmark_frames})
            frames = reader.frames()
            measured = 0
            while True:
                await self._checkpoint(execution)
                packet = await asyncio.to_thread(next, frames, None)
                if packet is None:
                    break
                collector.decoded()
                if packet.index % stream.frame_stride:
                    collector.skipped()
                    continue
                frame_started = perf_counter()
                result = await self.manager.predict(stream.config, packet.image)
                warmup = collector.frames["warmup"] < stream.warmup_frames
                gpu = {"device": result.metrics.device_name or stream.config.device, "allocated_mb": result.metrics.gpu_memory_mb, "reserved_mb": result.metrics.gpu_reserved_mb, "peak_allocated_mb": result.metrics.gpu_peak_memory_mb}
                if warmup:
                    collector.add({"decode_ms": packet.decode_ms, "preprocess_ms": result.metrics.preprocess_ms, "inference_ms": result.metrics.inference_ms, "postprocess_ms": result.metrics.postprocess_ms, "total_frame_ms": (perf_counter() - frame_started) * 1000}, warmup=True, rendered=False, gpu=gpu)
                    continue
                render_started = perf_counter()
                canvas = await asyncio.to_thread(self.renderers.render, packet.image, result)
                render_ms = (perf_counter() - render_started) * 1000
                url = await self._save_preview(execution.run_id, node_id, canvas, packet.index)
                total_frame_ms = (perf_counter() - frame_started) * 1000
                frame_result = VideoFrameResult(frame_index=packet.index, timestamp_seconds=packet.timestamp_seconds, result=result)
                frame_metrics = {"decode_ms": packet.decode_ms, "preprocess_ms": result.metrics.preprocess_ms, "inference_ms": result.metrics.inference_ms, "postprocess_ms": result.metrics.postprocess_ms, "render_ms": render_ms, "publish_ms": 0.0, "total_frame_ms": total_frame_ms, "model_fps": 1000 / result.metrics.inference_ms if result.metrics.inference_ms else 0, "effective_fps": 1000 / total_frame_ms if total_frame_ms else 0}
                event = {"type": "video_frame", "event": "frame_processed", "node_id": node_id, "frame": packet.index, "frame_index": packet.index, "frame_count": stream.asset.frame_count, "timestamp_seconds": packet.timestamp_seconds, "duration_seconds": stream.asset.duration_seconds, "progress": (packet.index + 1) / max(stream.asset.frame_count or 1, 1), "fps": stream.asset.fps, "url": url, "preview_url": url, "model": model, "frame_result": frame_result.model_dump(mode="json"), "result": result.model_dump(mode="json"), "metrics": frame_metrics, "object_count": len(getattr(result, "detections", getattr(result, "masks", []))), "gpu": gpu}
                publish_started = perf_counter()
                await self.publish(execution, event)
                frame_metrics["publish_ms"] = (perf_counter() - publish_started) * 1000
                collector.add(frame_metrics, warmup=False, rendered=True, gpu=gpu)
                measured += 1
                if stream.benchmark_frames and measured >= stream.benchmark_frames:
                    break
        except InterruptedError:
            final_state = "cancelled"
            raise
        except Exception as exc:
            final_state = "failed"
            await self.publish(execution, {"type": "video_lifecycle", "event": "video_failed", "node_id": node_id, "error": str(exc)})
            raise
        finally:
            await asyncio.to_thread(reader.close)
            report = collector.report(final_state)
            await asyncio.to_thread(self._persist_benchmark, execution.run_id, report)
            await self.publish(execution, {"type": "benchmark", "event": "benchmark_completed", "node_id": node_id, "benchmark": report, "benchmark_url": f"/api/v1/benchmarks/{execution.run_id}"})
        await self.publish(execution, {"type": "video_lifecycle", "event": "video_completed", "node_id": node_id, "frames": report["frames"]})
        return report

    def _persist_benchmark(self, run_id: str, report: dict[str, Any]) -> None:
        target = self.benchmark_dir / f"{run_id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(target)

    async def _save_preview(self, run_id: str, node_id: str, image, frame: int | None = None, label: str | None = None) -> str:
        suffix = f"_slot_{frame % self.preview_retention:02d}" if frame is not None else f"_{label}" if label else ""
        filename = f"{run_id}_{node_id}{suffix}.jpg"
        target = self.preview_dir / filename
        temporary = target.with_name(f".{target.stem}.tmp.jpg")
        success = await asyncio.to_thread(cv2.imwrite, str(temporary), image)
        if not success:
            raise RuntimeError("Could not write preview")
        await asyncio.to_thread(temporary.replace, target)
        version = f"?frame={frame}" if frame is not None else ""
        return f"/previews/{filename}{version}"
