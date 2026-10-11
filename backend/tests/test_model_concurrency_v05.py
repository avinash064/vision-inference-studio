from __future__ import annotations

import asyncio
import io
import threading
import time

from inference_studio.adapters import AdapterRegistry, LoadedModel, ModelAdapter, ModelConfig
from inference_studio.models import ModelManager
from inference_studio.results import DetectionResult


class IsolatedAdapter(ModelAdapter):
    name = "yolo"

    def __init__(self):
        self.loads, self.active, self.max_active = 0, 0, 0
        self.requests = []
        self.lock = threading.Lock()

    def load(self, key, weight_path):
        self.loads += 1
        return LoadedModel({}, key, weight_path)

    def preprocess(self, model, request, image):
        return image

    def infer(self, model, request, image):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.requests.append((request.confidence, request.text_prompt))
        time.sleep(.03)
        with self.lock:
            self.active -= 1
        return None

    def postprocess(self, model, request, output):
        return DetectionResult()


async def test_same_model_loads_once_and_requests_do_not_cross_talk(model_registry, stores, registered_files):
    weights, _ = stores
    weight, _ = registered_files
    adapter = IsolatedAdapter()
    manager = ModelManager(model_registry, weights, AdapterRegistry([adapter]))
    base = dict(family="yolo", version="v11", size="n", weight_id=weight.weight_id)
    first = ModelConfig(**base, confidence=.1, text_prompt="first")
    second = ModelConfig(**base, confidence=.9, text_prompt="second")
    await asyncio.gather(manager.predict(first, object()), manager.predict(second, object()))
    assert adapter.loads == 1
    assert adapter.max_active == 1
    assert sorted(adapter.requests) == [(.1, "first"), (.9, "second")]
    assert (await manager.load(first)).key == first.model_key()


async def test_two_dino_nodes_share_one_model_handle(model_registry, stores):
    weights, _ = stores
    weight = weights.save(io.BytesIO(b"synthetic dino state"), "model.pth", "dinov3")
    adapter = IsolatedAdapter()
    adapter.name = "dinov3"
    manager = ModelManager(model_registry, weights, AdapterRegistry([adapter]))
    first = ModelConfig(family="dinov3", version="official", size="vit-b", weight_id=weight.weight_id, confidence=.1)
    second = first.model_copy(update={"confidence": .9})
    one, two = await asyncio.gather(manager.load(first), manager.load(second))
    assert one is two
    assert adapter.loads == 1
    assert first.model_key() == second.model_key()
