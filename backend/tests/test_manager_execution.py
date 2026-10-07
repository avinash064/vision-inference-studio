from __future__ import annotations

import asyncio

from inference_studio.adapters import AdapterRegistry, LoadedModel, ModelAdapter, ModelConfig
from inference_studio.graph.executor import ExecutionEngine, RunState
from inference_studio.graph.validator import WorkflowValidator
from inference_studio.models import ModelManager
from inference_studio.nodes import NodeRegistry
from inference_studio.results import Detection, DetectionResult, InferenceMetrics

from .test_graph import valid_workflow


class FakeAdapter(ModelAdapter):
    name = "yolo"

    def __init__(self):
        self.loads = 0
        self.unloads = 0

    def load(self, config, weight_path):
        self.loads += 1
        return LoadedModel(handle={"ready": True}, config=config, weight_path=weight_path)

    def preprocess(self, model, input_data):
        return input_data

    def infer(self, model, tensor):
        return tensor

    def postprocess(self, model, output):
        return DetectionResult()

    def predict(self, model, image):
        return DetectionResult(detections=[Detection(bbox=[4, 5, 30, 35], class_id=0, label="object", confidence=.91)], metrics=InferenceMetrics(inference_ms=1.2, total_ms=1.4))

    def unload(self, model):
        self.unloads += 1


async def wait_terminal(engine, run_id):
    for _ in range(100):
        if engine.get(run_id).state in {RunState.COMPLETED, RunState.ERROR, RunState.STOPPED}:
            return
        await asyncio.sleep(.01)
    raise AssertionError("execution did not finish")


async def test_model_manager_caches_and_unloads(model_registry, stores, registered_files):
    weights, _ = stores
    weight, _ = registered_files
    fake = FakeAdapter()
    manager = ModelManager(model_registry, weights, AdapterRegistry([fake]))
    config = ModelConfig(family="yolo", version="v11", size="n", weight_id=weight.weight_id)
    first = await manager.load(config)
    second = await manager.load(config)
    assert first is second and fake.loads == 1
    await manager.unload(config)
    assert fake.unloads == 1 and manager.status()[0].state == "UNLOADED"


async def test_execution_serializes_results_and_selects_detection_renderer(tmp_path, model_registry, stores, registered_files):
    weights, assets = stores
    weight, asset = registered_files
    fake = FakeAdapter()
    manager = ModelManager(model_registry, weights, AdapterRegistry([fake]))
    validator = WorkflowValidator(NodeRegistry(), model_registry, weights, assets)
    engine = ExecutionEngine(validator, assets, manager, tmp_path / "previews")
    execution = engine.start(valid_workflow(weight.weight_id, asset.asset_id))
    await wait_terminal(engine, execution.run_id)
    assert execution.state == RunState.COMPLETED
    result_events = [event for event in execution.events if event["type"] == "inference_result"]
    assert result_events[0]["result"]["detections"][0]["label"] == "object"
    preview = next(event for event in execution.events if event["type"] == "preview")
    assert (tmp_path / "previews" / preview["url"].split("/")[-1]).is_file()
