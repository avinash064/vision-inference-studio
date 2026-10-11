from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from inference_studio.features.extractors import LoadedFeatureModel, RFDetrArchBFeatureExtractor
from inference_studio.features.ops import difference, load_mahalanobis_geometry, mahalanobis_difference
from inference_studio.features.store import FeatureStore, artifact_nbytes


class FakeVendor:
    """Stands in for vendor/rsd_v1_extract_rfdetr_features.py and counts backbone/head passes."""

    RESOLUTION = 504
    FEATURE_SPATIAL = 42

    def __init__(self):
        self.backbone_calls = 0
        self.head_calls = 0
        self.preprocessed = []

    def load_frozen_model(self, checkpoint_path, device):
        return SimpleNamespace(device=torch.device(device), checkpoint=checkpoint_path)

    def preprocess_image(self, image, resolution, device=None):
        self.preprocessed.append((image.shape, resolution))
        return torch.zeros(1, 3, resolution, resolution), image.shape[:2]

    def extract_backbone_feature(self, frozen, tensor):
        self.backbone_calls += 1
        src = torch.arange(256 * 42 * 42, dtype=torch.float32).reshape(1, 256, 42, 42)
        return [src], [torch.zeros(1, 42, 42, dtype=torch.bool)], [torch.ones(1, 256, 42, 42)]

    def run_heads_from_srcs(self, frozen, srcs, masks, poss, image_size):
        self.head_calls += 1
        return {"pred_masks": torch.zeros(1, 3, 126, 126), "image_size": image_size, "srcs_id": id(srcs[0])}


@pytest.fixture
def fake(monkeypatch):
    vendor = FakeVendor()
    monkeypatch.setattr(RFDetrArchBFeatureExtractor, "_vendor", staticmethod(lambda: vendor))
    return vendor


def _loaded(extractor: RFDetrArchBFeatureExtractor) -> LoadedFeatureModel:
    key = extractor.model_key("wgt_b", "cpu", "fp32")
    return extractor.load(key, Path("seglarge.pth"), "d" * 64)


def test_extracts_srcs0_as_42x42x256_with_504_patch12_signature(fake):
    extractor = RFDetrArchBFeatureExtractor()
    artifact = extractor.extract(_loaded(extractor), np.zeros((720, 1280, 3), np.uint8), "backbone_42")
    assert tuple(artifact.tensor.shape) == (42, 42, 256)
    src = fake.extract_backbone_feature(None, None)[0][0]
    assert torch.equal(artifact.tensor[7, 9], src[0, :, 7, 9])  # [1,256,42,42] -> [42,42,256]
    signature = artifact.metadata.signature
    assert (signature.extractor_version, signature.extractor_variant, signature.feature_stage) == ("architecture-b", "seglarge", "backbone_42")
    assert (signature.input_height, signature.patch_size, signature.stride, signature.checkpoint_sha256) == (504, 12, 12.0, "d" * 64)
    assert fake.preprocessed == [((720, 1280, 3), 504)]
    with pytest.raises(ValueError):
        extractor.extract(_loaded(extractor), np.zeros((8, 8, 3), np.uint8), "fine_80")


def test_one_forward_is_shared_with_the_future_semantic_branch(fake):
    extractor = RFDetrArchBFeatureExtractor()
    loaded = _loaded(extractor)
    artifact = extractor.extract(loaded, np.zeros((64, 64, 3), np.uint8), "backbone_42")
    forward = artifact.shared_forward
    assert forward is not None and len(forward.srcs) == len(forward.masks) == len(forward.poss) == 1
    first = extractor.semantic_outputs(loaded, forward)
    second = extractor.semantic_outputs(loaded, forward)
    assert first is second and first["srcs_id"] == id(forward.srcs[0]) and first["image_size"] == (504, 504)
    assert fake.backbone_calls == 1 and fake.head_calls == 1  # no second backbone pass
    assert artifact_nbytes(artifact) == artifact.metadata.memory_bytes + 256 * 42 * 42 * 4 * 2 + 42 * 42


def test_store_accounts_and_releases_shared_forward(fake):
    extractor = RFDetrArchBFeatureExtractor()
    artifact = extractor.extract(_loaded(extractor), np.zeros((64, 64, 3), np.uint8), "backbone_42")
    store = FeatureStore(max_entries=4, max_bytes=1 << 30)
    value = store.put(artifact, "run")
    assert store.stats()["bytes"] == artifact_nbytes(artifact)
    store.release(value)
    assert store.stats()["entries"] == 0 and store.stats()["bytes"] == 0


real = pytest.mark.skipif(
    not (os.getenv("VIS_ARCHB_WEIGHTS") and os.getenv("VIS_FEATURE_TEST_IMAGE")),
    reason="Set VIS_ARCHB_WEIGHTS and VIS_FEATURE_TEST_IMAGE (optional VIS_ARCHB_GEOMETRY) for the real Architecture B test",
)


@pytest.mark.integration
@real
def test_real_architecture_b_shape_identity_and_mahalanobis():
    path = Path(os.environ["VIS_ARCHB_WEIGHTS"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    extractor = RFDetrArchBFeatureExtractor()
    loaded = extractor.load(extractor.model_key("real", device, "fp32"), path, hashlib.sha256(path.read_bytes()).hexdigest())
    image = cv2.imread(os.environ["VIS_FEATURE_TEST_IMAGE"])
    reference = extractor.extract(loaded, image, "backbone_42")
    current = extractor.extract(loaded, image, "backbone_42")
    assert tuple(reference.tensor.shape) == (42, 42, 256) and reference.tensor.device.type == torch.device(device).type
    delta, l2 = difference(reference, current, "l2")
    assert float(delta.tensor.abs().max()) < 1e-3 and l2.statistics["max"] < 1e-3
    if os.getenv("VIS_ARCHB_GEOMETRY"):
        geometry = load_mahalanobis_geometry(os.environ["VIS_ARCHB_GEOMETRY"])
        identity = mahalanobis_difference(reference, current, geometry)
        assert identity.statistics["max"] - identity.statistics["min"] < 1e-2 * max(1.0, identity.statistics["max"])
