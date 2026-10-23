from __future__ import annotations

import asyncio

from inference_studio.adapters import AdapterRegistry, LoadedModel, ModelAdapter, ModelConfig
from inference_studio.graph.executor import Execution, ExecutionEngine, RunState
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
        return LoadedModel(handle={"ready": True}, key=config, weight_path=weight_path)

    def preprocess(self, model, request, input_data):
        return input_data

    def infer(self, model, request, tensor):
        return tensor

    def postprocess(self, model, request, output):
        return DetectionResult()

    def predict(self, model, request, image):
        return DetectionResult(detections=[Detection(bbox=[4, 5, 30, 35], class_id=0, label="object", confidence=.91)], metrics=InferenceMetrics(inference_ms=1.2, total_ms=1.4))

    def unload(self, model):
        self.unloads += 1


class FakeWeightlessAdapter(FakeAdapter):
    name = "tesseract"


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
    changed_threshold = config.model_copy(update={"confidence": .8})
    second = await manager.load(changed_threshold)
    assert first is second and fake.loads == 1
    assert second.key == config.model_key()
    await manager.unload(config)
    assert fake.unloads == 1 and manager.status()[0].state == "UNLOADED"


async def test_model_manager_loads_weightless_engine_without_weight_lookup(model_registry, stores):
    weights, _ = stores
    fake = FakeWeightlessAdapter()
    manager = ModelManager(model_registry, weights, AdapterRegistry([fake]))
    config = ModelConfig(family="tesseract", version="5", size="system", ocr_language="eng")
    loaded = await manager.load(config)
    assert fake.loads == 1 and loaded.weight_path == __import__("pathlib").Path()
    assert manager.status()[0].weight_id == ""


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
    assert preview["event"] == "preview_ready"
    assert preview["url"] == preview["prediction_url"]
    for url in (preview["original_url"], preview["prediction_url"]):
        assert (tmp_path / "previews" / url.split("/")[-1]).is_file()
    lifecycle = [event["event"] for event in execution.events]
    assert lifecycle[0] == "workflow_started"
    assert lifecycle.count("node_waiting") == 3
    assert "inference_completed" in lifecycle
    assert lifecycle[-1] == "workflow_completed"


async def test_live_subscriber_receives_terminal_event(tmp_path, model_registry, stores, registered_files):
    weights, assets = stores
    weight, asset = registered_files
    manager = ModelManager(model_registry, weights, AdapterRegistry([FakeAdapter()]))
    validator = WorkflowValidator(NodeRegistry(), model_registry, weights, assets)
    engine = ExecutionEngine(validator, assets, manager, tmp_path / "previews")
    execution = Execution(run_id="run_live", workflow=valid_workflow(weight.weight_id, asset.asset_id), state=RunState.RUNNING)
    engine.executions[execution.run_id] = execution
    events = engine.subscribe(execution.run_id)

    waiting = asyncio.create_task(anext(events))
    await asyncio.sleep(0)
    await engine.publish(execution, {"type": "node_state", "event": "node_completed", "state": "success"})
    assert (await waiting)["event"] == "node_completed"

    execution.state = RunState.COMPLETED
    await engine.publish(execution, {"type": "run_state", "event": "workflow_completed", "state": execution.state})
    assert (await anext(events))["event"] == "workflow_completed"
    try:
        await anext(events)
        assert False, "terminal event must close the subscription"
    except StopAsyncIteration:
        pass
