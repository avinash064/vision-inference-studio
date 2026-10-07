from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inference_studio.adapters import AdapterRegistry, LoadedModel, ModelConfig
from inference_studio.adapters.rfdetr import RfDetrAdapter
from inference_studio.adapters.yolo import YoloAdapter
from inference_studio.results import DetectionResult


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def loaded(family: str = "rfdetr") -> LoadedModel:
    config = ModelConfig(family=family, version="v1" if family == "rfdetr" else "v11", size="large" if family == "rfdetr" else "n", weight_id="test")
    return LoadedModel(handle=SimpleNamespace(class_names={4: "fallback"}), config=config, weight_path=Path("test.pth"))


def test_adapter_registry_resolves_model_specific_code_once():
    registry = AdapterRegistry()
    assert registry.get("rfdetr").name == "rfdetr"
    assert registry.get("yolo").name == "yolo"


def test_rfdetr_normalizes_supervision_detection_and_rgb_preprocessing():
    adapter = RfDetrAdapter()
    prediction = SimpleNamespace(xyxy=np.asarray([[10, 20, 80, 90]], dtype=float), class_id=np.asarray([4]), confidence=np.asarray([.87]), data={"class_name": np.asarray(["skip_car"])})
    result = adapter.postprocess(loaded(), prediction)
    assert result == DetectionResult.model_validate({"result_type": "detections", "detections": [{"bbox": [10, 20, 80, 90], "class_id": 4, "label": "skip_car", "confidence": .87}]})
    bgr = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    assert adapter.preprocess(loaded(), bgr).tolist() == [[[3, 2, 1]]]


def test_yolo_normalizes_ultralytics_detection():
    adapter = YoloAdapter()
    output = SimpleNamespace(names={2: "skip_car"}, boxes=SimpleNamespace(xyxy=FakeTensor([[1, 2, 30, 40]]), cls=FakeTensor([2]), conf=FakeTensor([.93])))
    result = adapter.postprocess(loaded("yolo"), output)
    assert result.result_type == "detections"
    assert result.detections[0].model_dump() == {"bbox": [1.0, 2.0, 30.0, 40.0], "class_id": 2, "label": "skip_car", "confidence": .93}
