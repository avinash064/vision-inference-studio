from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

from inference_studio.adapters import ModelConfig
from inference_studio.adapters.dinov3 import DinoV3Adapter
from inference_studio.runtime import FeatureArtifact
from inference_studio.runtime.features import difference, normalize


pytestmark = pytest.mark.integration


def test_real_dinov3_identity_and_current_image_when_configured():
    weight = os.getenv("VIS_DINO_WEIGHTS")
    reference_path = os.getenv("VIS_DINO_TEST_REFERENCE")
    current_path = os.getenv("VIS_DINO_TEST_CURRENT")
    if not weight or not reference_path:
        pytest.skip("Set VIS_DINO_WEIGHTS and VIS_DINO_TEST_REFERENCE to run real DINOv3 integration")
    reference = cv2.imread(reference_path)
    current = cv2.imread(current_path) if current_path else reference.copy()
    assert reference is not None and current is not None
    device = os.getenv("VIS_TEST_DEVICE", "cuda:0")
    config = ModelConfig(family="dinov3", version="official", size=os.getenv("VIS_DINO_SIZE", "vit-b"), weight_id="integration", device=device, precision="fp16" if device.startswith("cuda") else "fp32")
    adapter = DinoV3Adapter()
    model = adapter.load(config.model_key(), Path(weight))
    try:
        model.checkpoint_sha256 = os.getenv("VIS_DINO_SHA256", "integration")
        reference_feature = adapter.predict(model, config.inference_request(), reference)
        identity_feature = adapter.predict(model, config.inference_request(), reference)
        current_feature = adapter.predict(model, config.inference_request(), current)
        assert all(isinstance(value, FeatureArtifact) for value in (reference_feature, identity_feature, current_feature))
        _, identity = difference(normalize(reference_feature), normalize(identity_feature), "cosine")
        _, changed = difference(normalize(reference_feature), normalize(current_feature), "cosine")
        assert identity.statistics["mean"] < 1e-5
        assert identity.statistics["p99"] < 1e-4
        if current_path and Path(current_path).resolve() != Path(reference_path).resolve():
            assert changed.statistics["max"] >= identity.statistics["max"]
    finally:
        adapter.unload(model)
