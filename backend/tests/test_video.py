from __future__ import annotations

import asyncio
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from inference_studio.adapters import AdapterRegistry, LoadedModel, ModelAdapter
from inference_studio.graph import Edge, Workflow
from inference_studio.graph.executor import ExecutionEngine, RunState
from inference_studio.graph.validator import WorkflowValidator
from inference_studio.models import ModelManager
from inference_studio.nodes import NodeInstance, NodeRegistry
from inference_studio.results import Detection, DetectionResult, InferenceMetrics, VideoFrameResult
from inference_studio.video import VideoMetricsCollector, VideoReader

from .test_graph import model_config


class VideoAdapter(ModelAdapter):
    name = "yolo"

    def __init__(self, delay: float = 0) -> None:
        self.delay = delay
        self.loads = 0
        self.predictions = 0

    def load(self, config, weight_path):
        self.loads += 1
        return LoadedModel(handle={}, config=config, weight_path=weight_path)

    def preprocess(self, model, input_data):
        return input_data

    def infer(self, model, tensor):
        return tensor

    def postprocess(self, model, output):
        return DetectionResult()

    def predict(self, model, image):
        if self.delay:
            time.sleep(self.delay)
        self.predictions += 1
        return DetectionResult(detections=[Detection(bbox=[1, 2, 12, 14], class_id=0, label="vehicle", confidence=.9)], metrics=InferenceMetrics(preprocess_ms=1, inference_ms=4, postprocess_ms=1, total_ms=6))


def make_video(path: Path, frames: int = 24, fps: float = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, 1] = (index * 4) % 256
        cv2.putText(image, str(index), (5, 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        writer.write(image)
    writer.release()
    return path


def register_video(assets, tmp_path: Path, frames: int = 24):
    path = make_video(tmp_path / "source.avi", frames=frames)
    with path.open("rb") as source:
        return assets.save(source, "camera.avi", "video")


def video_workflow(weight_id: str, asset_id: str, *, stride: int = 2, warmup: int = 1, benchmark: int = 5) -> Workflow:
    return Workflow(nodes=[
        NodeInstance(id="video", type="video_source", config={"asset_id": asset_id, "frame_stride": stride, "warmup_frames": warmup, "benchmark_frames": benchmark}),
        NodeInstance(id="model", type="model", config=model_config(weight_id)),
        NodeInstance(id="preview", type="video_preview"),
    ], edges=[
        Edge(id="source-model", source="video", source_handle="video", target="model", target_handle="input"),
        Edge(id="model-preview", source="model", source_handle="output", target="preview", target_handle="result"),
    ])


async def wait_for(engine, run_id: str, predicate, attempts: int = 500):
    for _ in range(attempts):
        execution = engine.get(run_id)
        if predicate(execution):
            return execution
        await asyncio.sleep(.01)
    raise AssertionError("execution condition timed out")


def test_video_metadata_sequential_reader_seek_and_timestamps(tmp_path, stores):
    _, assets = stores
    record = register_video(assets, tmp_path, frames=8)
    assert record.filename == "camera.avi"
    assert record.codec and record.size_bytes > 0
    assert (record.width, record.height, record.frame_count, record.fps) == (64, 48, 8, 10)
    assert record.duration_seconds == pytest.approx(.8)
    with VideoReader(record.path) as reader:
        first = list(reader.frames())
        assert [frame.index for frame in first] == list(range(8))
        assert first[5].timestamp_seconds == pytest.approx(.5)
        reader.seek(3)
        assert next(reader.frames()).index == 3


def test_video_frame_result_and_benchmark_aggregation():
    result = VideoFrameResult(frame_index=25, timestamp_seconds=2.5, result=DetectionResult(detections=[Detection(bbox=[1, 2, 3, 4], class_id=1, label="car", confidence=.8)]))
    assert result.model_dump()["result"]["detections"][0]["label"] == "car"
    collector = VideoMetricsCollector({"fps": 10}, {"family": "yolo"}, warmup_frames=1)
    collector.decoded(); collector.add({"inference_ms": 100}, warmup=True, rendered=False, gpu={})
    for value in (10, 20, 30, 40):
        collector.decoded(); collector.add({"inference_ms": value, "total_frame_ms": value + 5}, warmup=False, rendered=True, gpu={"peak_allocated_mb": value})
    report = collector.report("completed")
    assert report["latency_ms"]["inference_ms"]["median"] == 25
    assert report["latency_ms"]["inference_ms"]["p95"] == pytest.approx(38.5)
    assert report["frames"] == {"decoded": 5, "inferred": 5, "skipped": 0, "rendered": 4, "warmup": 1}
    assert report["gpu"]["peak_allocated_mb"] == 40


async def test_video_execution_stride_model_reuse_and_bounded_previews(tmp_path, model_registry, stores, registered_files):
    weights, assets = stores
    weight, _ = registered_files
    video = register_video(assets, tmp_path, frames=24)
    adapter = VideoAdapter()
    manager = ModelManager(model_registry, weights, AdapterRegistry([adapter]))
    engine = ExecutionEngine(WorkflowValidator(NodeRegistry(), model_registry, weights, assets), assets, manager, tmp_path / "previews", tmp_path / "benchmarks", preview_retention=3)
    workflow = video_workflow(weight.weight_id, video.asset_id)

    first = engine.start(workflow)
    await wait_for(engine, first.run_id, lambda item: item.state in {RunState.COMPLETED, RunState.ERROR})
    assert first.state == RunState.COMPLETED
    assert adapter.loads == 1 and adapter.predictions == 6  # one warm-up + five measured
    benchmark = next(event["benchmark"] for event in first.events if event.get("type") == "benchmark")
    assert benchmark["frames"]["inferred"] == 6
    assert benchmark["frames"]["skipped"] == 5
    latest = next(event for event in first.events if event.get("type") == "video_frame")
    assert latest["frame_index"] == 10 and latest["timestamp_seconds"] == pytest.approx(1.0)
    assert len(list((tmp_path / "previews").glob(f"{first.run_id}_preview_slot_*.jpg"))) <= 3
    assert (tmp_path / "benchmarks" / f"{first.run_id}.json").is_file()

    second = engine.start(workflow)
    await wait_for(engine, second.run_id, lambda item: item.state in {RunState.COMPLETED, RunState.ERROR})
    assert second.state == RunState.COMPLETED and adapter.loads == 1
    assert any(event.get("event") == "model_reused" for event in second.events)


async def test_video_pause_resume_and_stop_at_frame_boundaries(tmp_path, model_registry, stores, registered_files):
    weights, assets = stores
    weight, _ = registered_files
    video = register_video(assets, tmp_path, frames=80)
    adapter = VideoAdapter(delay=.015)
    manager = ModelManager(model_registry, weights, AdapterRegistry([adapter]))
    engine = ExecutionEngine(WorkflowValidator(NodeRegistry(), model_registry, weights, assets), assets, manager, tmp_path / "previews", tmp_path / "benchmarks", preview_retention=2)
    execution = engine.start(video_workflow(weight.weight_id, video.asset_id, stride=1, warmup=0, benchmark=0))
    await wait_for(engine, execution.run_id, lambda item: any(event.get("type") == "video_frame" for event in item.events))
    await engine.pause(execution.run_id)
    await wait_for(engine, execution.run_id, lambda item: item.state == RunState.PAUSED)
    paused_predictions = adapter.predictions
    await asyncio.sleep(.06)
    assert adapter.predictions == paused_predictions
    await engine.resume(execution.run_id)
    await wait_for(engine, execution.run_id, lambda item: adapter.predictions > paused_predictions)
    await engine.stop(execution.run_id)
    await wait_for(engine, execution.run_id, lambda item: item.state == RunState.CANCELLED)
    assert any(event.get("event") == "workflow_paused" for event in execution.events)
    assert any(event.get("event") == "workflow_resumed" for event in execution.events)
    assert any(event.get("event") == "cancel_requested" for event in execution.events)
    assert any(event.get("event") == "workflow_cancelled" for event in execution.events)
