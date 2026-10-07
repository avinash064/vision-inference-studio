from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from inference_studio.adapters import ModelConfig
from inference_studio.assets import AssetStore
from inference_studio.models import ModelManager
from inference_studio.nodes.runtime import ImageValue, ModelOutput, VideoModelStream
from inference_studio.renderers import RendererRegistry

from .schema import Workflow
from .validator import WorkflowValidator


class RunState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass
class Execution:
    run_id: str
    workflow: Workflow
    state: RunState = RunState.QUEUED
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    pause_gate: asyncio.Event = field(default_factory=asyncio.Event)
    stop_requested: bool = False

    def __post_init__(self) -> None:
        self.pause_gate.set()


class ExecutionEngine:
    def __init__(self, validator: WorkflowValidator, assets: AssetStore, manager: ModelManager, preview_dir: Path) -> None:
        self.validator, self.assets, self.manager = validator, assets, manager
        self.preview_dir = preview_dir
        self.renderers = RendererRegistry()
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.executions: dict[str, Execution] = {}

    def start(self, workflow: Workflow) -> Execution:
        validation = self.validator.validate(workflow)
        if not validation.valid:
            raise ValueError("; ".join(issue.message for issue in validation.issues))
        execution = Execution(run_id=f"run_{uuid.uuid4().hex}", workflow=workflow)
        self.executions[execution.run_id] = execution
        asyncio.create_task(self._execute(execution, validation.order))
        return execution

    async def publish(self, execution: Execution, event: dict[str, Any]) -> None:
        event = {"run_id": execution.run_id, **event}
        execution.events.append(event)
        for queue in list(execution.subscribers):
            await queue.put(event)

    async def subscribe(self, run_id: str):
        execution = self.get(run_id)
        queue: asyncio.Queue = asyncio.Queue()
        execution.subscribers.append(queue)
        terminal = {RunState.COMPLETED, RunState.ERROR, RunState.STOPPED}
        try:
            # Snapshot after registering the queue: later events cannot fall into
            # the gap between replay and live delivery.
            for event in list(execution.events):
                yield event
                if event.get("type") == "run_state" and event.get("state") in terminal:
                    return
            while True:
                event = await queue.get()
                yield event
                if event.get("type") == "run_state" and event.get("state") in terminal:
                    return
        finally:
            execution.subscribers.remove(queue)

    def get(self, run_id: str) -> Execution:
        try:
            return self.executions[run_id]
        except KeyError as exc:
            raise ValueError(f"Unknown run: {run_id}") from exc

    async def pause(self, run_id: str) -> None:
        execution = self.get(run_id)
        execution.state = RunState.PAUSED
        execution.pause_gate.clear()
        await self.publish(execution, {"type": "run_state", "event": "workflow_paused", "state": execution.state})

    async def resume(self, run_id: str) -> None:
        execution = self.get(run_id)
        execution.state = RunState.RUNNING
        execution.pause_gate.set()
        await self.publish(execution, {"type": "run_state", "event": "workflow_resumed", "state": execution.state})

    async def stop(self, run_id: str) -> None:
        execution = self.get(run_id)
        execution.stop_requested = True
        execution.pause_gate.set()

    async def _checkpoint(self, execution: Execution) -> None:
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
                incoming = {edge.target_handle: values[edge.source] for edge in execution.workflow.edges if edge.target == node_id}
                values[node_id] = await self._run_node(execution, node, incoming)
                await self.publish(execution, {"type": "node_state", "event": "node_completed", "node_id": node_id, "state": "success", "duration_ms": (perf_counter() - node_started) * 1000})
            execution.state = RunState.COMPLETED
            await self.publish(execution, {"type": "run_state", "event": "workflow_completed", "state": execution.state})
        except InterruptedError:
            execution.state = RunState.STOPPED
            await self.publish(execution, {"type": "run_state", "event": "workflow_cancelled", "state": execution.state})
        except Exception as exc:
            execution.state = RunState.ERROR
            if current_node_id is not None:
                await self.publish(execution, {"type": "node_state", "event": "node_failed", "node_id": current_node_id, "state": "error", "error": str(exc)})
            await self.publish(execution, {"type": "run_state", "event": "workflow_failed", "state": execution.state, "error": str(exc)})

    async def _run_node(self, execution: Execution, node, inputs: dict[str, Any]):
        if node.type in {"image_source", "video_source"}:
            asset = self.assets.get(str(node.config["asset_id"]))
            if node.type == "video_source":
                return asset
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
        if node.type == "model":
            config = ModelConfig.model_validate(node.config)
            value = inputs["input"]
            if hasattr(value, "kind") and value.kind == "video":
                return VideoModelStream(asset=value, config=config)
            await self.manager.load(config)
            await self.publish(execution, {"type": "node_state", "event": "node_started", "node_id": node.id, "state": "running"})
            result = await self.manager.predict(config, value.image)
            await self.publish(execution, {"type": "inference_result", "event": "inference_completed", "node_id": node.id, "model": config.model_dump(mode="json"), "result": result.model_dump(mode="json")})
            return ModelOutput(image=value.image, result=result, config=config)
        if node.type == "video_preview":
            return await self._run_video(execution, node.id, inputs["result"])
        output = inputs["result"]
        expected = {"detection_preview": "detections", "feature_preview": ("feature_map", "embedding"), "tensor_preview": "tensor"}[node.type]
        if output.result.result_type not in ((expected,) if isinstance(expected, str) else expected):
            raise ValueError(f"{node.type} cannot render {output.result.result_type}")
        canvas = self.renderers.render(output.image, output.result)
        original_url = await self._save_preview(execution.run_id, node.id, output.image, label="original")
        prediction_url = await self._save_preview(execution.run_id, node.id, canvas, label="prediction")
        await self.publish(execution, {"type": "preview", "event": "preview_ready", "node_id": node.id, "url": prediction_url, "original_url": original_url, "prediction_url": prediction_url, "model": output.config.model_dump(mode="json"), "result": output.result.model_dump(mode="json")})
        return {"original_url": original_url, "prediction_url": prediction_url}

    async def _run_video(self, execution: Execution, node_id: str, stream: VideoModelStream):
        capture = cv2.VideoCapture(stream.asset.path)
        if not capture.isOpened():
            raise RuntimeError("Video asset can no longer be decoded")
        frame_number = 0
        try:
            while True:
                await self._checkpoint(execution)
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok:
                    break
                result = await self.manager.predict(stream.config, frame)
                canvas = self.renderers.render(frame, result)
                url = await self._save_preview(execution.run_id, node_id, canvas, frame_number)
                await self.publish(execution, {"type": "video_frame", "event": "video_frame_completed", "node_id": node_id, "frame": frame_number, "frame_count": stream.asset.frame_count, "fps": stream.asset.fps, "url": url, "model": stream.config.model_dump(mode="json"), "result": result.model_dump(mode="json"), "metrics": result.metrics.model_dump(), "object_count": len(getattr(result, "detections", []))})
                frame_number += 1
        finally:
            capture.release()
        return {"frames": frame_number}

    async def _save_preview(self, run_id: str, node_id: str, image, frame: int | None = None, label: str | None = None) -> str:
        suffix = f"_{frame:08d}" if frame is not None else f"_{label}" if label else ""
        filename = f"{run_id}_{node_id}{suffix}.jpg"
        target = self.preview_dir / filename
        success = await asyncio.to_thread(cv2.imwrite, str(target), image)
        if not success:
            raise RuntimeError("Could not write preview")
        return f"/previews/{filename}"
