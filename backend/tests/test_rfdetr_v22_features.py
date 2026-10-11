from __future__ import annotations

import asyncio
import hashlib
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from inference_studio.adapters import AdapterRegistry
from inference_studio.features.extractors import FeatureExtractorRegistry, LoadedFeatureModel, RFDetrV22FeatureExtractor
from inference_studio.features.extractors import rfdetr_v22
from inference_studio.features.ops import activation_xor, difference
from inference_studio.models import ModelManager


class FakeBoundary:
    """Stands in for the vendored RFDETRBoundaryFeatures: returns levels finest -> coarsest, [B,C,H,W]."""

    def __init__(self):
        self.calls = 0
        self.last_preprocess_ms = 1.5
        self.level_metadata = [{"shape": [80, 80]}, {"shape": [20, 20]}]

    def __call__(self, image):
        self.calls += 1
        fine = torch.arange(384 * 80 * 80, dtype=torch.float32).reshape(1, 384, 80, 80)
        coarse = torch.arange(384 * 20 * 20, dtype=torch.float32).reshape(1, 384, 20, 20) * -1
        return [fine, coarse]


def _loaded(handle=None) -> LoadedFeatureModel:
    extractor = RFDetrV22FeatureExtractor()
    return LoadedFeatureModel(handle or FakeBoundary(), extractor.model_key("wgt_x", "cpu", "fp32"), Path("x.pth"), "c" * 64)


def test_selects_levels_by_shape_and_converts_to_hwc():
    extractor, loaded = RFDetrV22FeatureExtractor(), _loaded()
    image = np.zeros((1080, 1920, 3), np.uint8)
    for stage, shape in extractor.STAGES.items():
        artifact = extractor.extract(loaded, image, stage)
        assert tuple(artifact.tensor.shape) == shape
        raw = loaded.handle(image)[0 if stage == "fine_80" else 1]
        assert torch.equal(artifact.tensor[3, 5], raw[0, :, 3, 5])  # [H,W,C] == permute(1,2,0) of batch 0
        signature = artifact.metadata.signature
        assert (signature.extractor_version, signature.extractor_variant, signature.feature_stage) == ("v2.2", "RFDETRLargeDeprecated", stage)
        assert signature.stride == {"fine_80": 7.0, "coarse_20": 28.0}[stage] and (signature.input_height, signature.input_width) == (560, 560)
        assert signature.preprocessing_signature == "rfdetr-v22-560-rgb01-imagenet" and signature.checkpoint_sha256 == "c" * 64
        assert artifact.metadata.source_width == 1920 and "feature_forward_ms" in artifact.metrics


def test_rejects_unknown_stage_and_missing_level():
    extractor = RFDetrV22FeatureExtractor()
    with pytest.raises(ValueError):
        extractor.extract(_loaded(), np.zeros((8, 8, 3), np.uint8), "backbone_42")

    class OnlyFine(FakeBoundary):
        def __call__(self, image):
            return super().__call__(image)[:1]

    with pytest.raises(RuntimeError, match="did not expose coarse_20"):
        extractor.extract(_loaded(OnlyFine()), np.zeros((8, 8, 3), np.uint8), "coarse_20")


def test_reference_and_current_share_one_model_and_stage_change_does_not_reload(monkeypatch, model_registry, stores):
    weights, _ = stores
    record = weights.save(io.BytesIO(b"rfdetr-v22-weights"), "rf-detr-large.pth", "rfdetr")
    handles = []

    def fake_runtime(weight_path, device):
        handles.append(FakeBoundary())
        return handles[-1]

    monkeypatch.setattr(rfdetr_v22, "_timed_boundary_features", fake_runtime)
    manager = ModelManager(model_registry, weights, AdapterRegistry(), FeatureExtractorRegistry([RFDetrV22FeatureExtractor()]))
    image = np.zeros((64, 64, 3), np.uint8)

    async def run():
        reference = await manager.extract_feature("rfdetr-v2.2", record.weight_id, image, "fine_80")
        current = await manager.extract_feature("rfdetr-v2.2", record.weight_id, image, "fine_80")
        coarse = await manager.extract_feature("rfdetr-v2.2", record.weight_id, image, "coarse_20")
        return reference, current, coarse

    reference, current, coarse = asyncio.run(run())
    assert manager.feature_load_count("rfdetr-v2.2", record.weight_id) == 1 and len(handles) == 1
    assert handles[0].calls == 3
    assert reference.metadata.signature.checkpoint_sha256 == hashlib.sha256(b"rfdetr-v22-weights").hexdigest()
    assert tuple(coarse.tensor.shape) == (20, 20, 384)


real = pytest.mark.skipif(
    not (os.getenv("VIS_RFDETR_V22_WEIGHTS") and os.getenv("VIS_FEATURE_TEST_IMAGE")),
    reason="Set VIS_RFDETR_V22_WEIGHTS and VIS_FEATURE_TEST_IMAGE for the real RF-DETR v2.2 feature test",
)


@pytest.mark.integration
@real
def test_real_v22_shapes_device_identity_and_localized_change():
    path = Path(os.environ["VIS_RFDETR_V22_WEIGHTS"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    extractor = RFDetrV22FeatureExtractor()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = extractor.load(extractor.model_key("real", device, "fp32"), path, sha)
    image = cv2.imread(os.environ["VIS_FEATURE_TEST_IMAGE"])
    changed = image.copy()
    h, w = image.shape[:2]
    changed[h // 3: h // 3 + h // 5, w // 3: w // 3 + w // 5] = (40, 200, 40)
    for stage, shape in extractor.STAGES.items():
        reference = extractor.extract(loaded, image, stage)
        current = extractor.extract(loaded, image, stage)
        assert tuple(reference.tensor.shape) == shape and reference.tensor.device.type == torch.device(device).type
        _, l2 = difference(reference, current, "l2")
        assert l2.statistics["max"] < 1e-3
        assert activation_xor(reference, current).statistics["max"] < 1e-3
        _, local = difference(reference, extractor.extract(loaded, changed, stage), "l2")
        assert local.statistics["max"] > l2.statistics["max"]
