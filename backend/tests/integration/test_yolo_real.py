from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

from inference_studio.adapters import ModelConfig
from inference_studio.adapters.yolo import YoloAdapter
from inference_studio.results import DetectionResult


pytestmark = pytest.mark.integration


def test_real_yolo_checkpoint_when_configured():
    weight = os.getenv("VIS_YOLO_WEIGHT")
    image_path = os.getenv("VIS_TEST_IMAGE")
    if not weight or not image_path:
        pytest.skip("Set VIS_YOLO_WEIGHT and VIS_TEST_IMAGE to run real YOLO integration")
    image = cv2.imread(image_path)
    assert image is not None
    adapter = YoloAdapter()
    config = ModelConfig(family="yolo", version="v11", size="custom", weight_id="integration", device=os.getenv("VIS_TEST_DEVICE", "cpu"), precision="fp16" if os.getenv("VIS_TEST_DEVICE", "cpu").startswith("cuda") else "fp32")
    model = adapter.load(config.model_key(), Path(weight))
    try:
        result = adapter.predict(model, config.inference_request(), image)
        assert isinstance(result, DetectionResult)
        assert result.metrics.total_ms > 0
    finally:
        adapter.unload(model)
