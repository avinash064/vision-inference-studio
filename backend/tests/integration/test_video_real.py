from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference_studio.adapters import AdapterRegistry, ModelConfig
from inference_studio.results import DetectionResult
from inference_studio.video import VideoReader


pytestmark = pytest.mark.integration


@pytest.mark.parametrize("family,weight_env,version,size", [
    ("yolo", "VIS_YOLO_WEIGHT", "v11", "custom"),
    ("rtdetr", "VIS_RTDETR_WEIGHT", "ultralytics", "custom"),
    ("rfdetr", "VIS_RFDETR_WEIGHT", "v1", "large"),
])
def test_real_video_frames_use_one_loaded_model(family, weight_env, version, size):
    video_path = os.getenv("VIS_TEST_VIDEO")
    weight_path = os.getenv(weight_env)
    if not video_path or not weight_path:
        pytest.skip(f"Set VIS_TEST_VIDEO and {weight_env} to run real {family} video integration")
    device = os.getenv("VIS_TEST_DEVICE", "cuda:0")
    target_frames = int(os.getenv("VIS_TEST_VIDEO_FRAMES", "100"))
    adapter = AdapterRegistry().get(family)
    config = ModelConfig(family=family, version=version, size=size, weight_id="integration", device=device, precision="fp16" if device.startswith("cuda") else "fp32", confidence=float(os.getenv("VIS_TEST_CONFIDENCE", ".3")))
    model = adapter.load(config.model_key(), Path(weight_path))
    processed = 0
    try:
        with VideoReader(video_path) as reader:
            for packet in reader.frames():
                result = adapter.predict(model, config.inference_request(), packet.image)
                assert isinstance(result, DetectionResult)
                processed += 1
                if processed >= target_frames:
                    break
    finally:
        adapter.unload(model)
    assert processed == target_frames
