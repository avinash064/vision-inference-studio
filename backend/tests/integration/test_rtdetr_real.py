from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

from inference_studio.adapters import ModelConfig
from inference_studio.adapters.rtdetr import RtDetrAdapter
from inference_studio.results import DetectionResult


pytestmark = pytest.mark.integration


def test_real_rtdetr_checkpoint_when_configured():
    weight = os.getenv("VIS_RTDETR_WEIGHT")
    image_path = os.getenv("VIS_TEST_IMAGE")
    if not weight or not image_path:
        pytest.skip("Set VIS_RTDETR_WEIGHT and VIS_TEST_IMAGE to run real RT-DETR integration")
    image = cv2.imread(image_path)
    assert image is not None
    device = os.getenv("VIS_TEST_DEVICE", "cpu")
    adapter = RtDetrAdapter()
    config = ModelConfig(family="rtdetr", version="ultralytics", size=os.getenv("VIS_RTDETR_SIZE", "custom"), weight_id="integration", device=device, precision="fp16" if device.startswith("cuda") else "fp32", confidence=float(os.getenv("VIS_TEST_CONFIDENCE", ".3")))
    model = adapter.load(config.model_key(), Path(weight))
    try:
        result = adapter.predict(model, config.inference_request(), image)
        assert isinstance(result, DetectionResult)
        assert result.metrics.total_ms > 0
        if device.startswith("cuda"):
            assert result.metrics.device_name
            assert result.metrics.gpu_peak_memory_mb > 0
    finally:
        adapter.unload(model)
