from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference_studio.adapters import InferenceRequest, LoadedModel, ModelKey
from inference_studio.adapters.dinov3 import DinoV3Adapter
from inference_studio.runtime import FeatureArtifact, FeatureMetadata, FeatureSpaceSignature, FeatureStore
from inference_studio.runtime.features import difference, normalize


def metadata(shape=(4, 5, 3), checkpoint="abc"):
    height, width, channels = shape
    space = FeatureSpaceSignature("dinov3", "official", "vit-b", checkpoint, "last_hidden_state", "processor", 64, 80, 16, height, width, channels, "default", "float32")
    return FeatureMetadata(shape, "float32", "cpu", height * width * channels * 4, 100, 120, height, width, channels, "last_hidden_state", "patch", "dinov3", "official", "vit-b", space)


def test_feature_identity_and_localized_change_metrics():
    base = torch.randn(4, 5, 3)
    original = FeatureArtifact(base, metadata(), {})
    normalized = normalize(original)
    _, identity = difference(normalized, normalized, "cosine")
    changed = normalized.tensor.clone()
    changed[2, 3] = -changed[2, 3]
    _, localized = difference(normalized, FeatureArtifact(changed, normalized.metadata, {}), "cosine")
    assert identity.statistics["max"] == pytest.approx(0, abs=1e-6)
    assert localized.values[2, 3].item() > localized.values.mean().item()
    assert {"median", "p90", "p95", "p99"} <= localized.statistics.keys()


def test_feature_space_mismatch_is_explicit():
    left = FeatureArtifact(torch.zeros(4, 5, 3), metadata(checkpoint="one"), {})
    right = FeatureArtifact(torch.zeros(4, 5, 3), metadata(checkpoint="two"), {})
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        difference(left, right, "l2")


def test_raw_absolute_l2_and_cosine_difference_modes():
    left_tensor = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    right_tensor = torch.tensor([[[3.0, 0.0], [0.0, -1.0]]])
    left = FeatureArtifact(left_tensor, metadata((1, 2, 2)), {})
    right = FeatureArtifact(right_tensor, metadata((1, 2, 2)), {})
    raw, raw_heatmap = difference(left, right, "raw")
    absolute, absolute_heatmap = difference(left, right, "absolute")
    l2, l2_heatmap = difference(left, right, "l2")
    cosine, cosine_heatmap = difference(left, right, "cosine")
    assert torch.equal(raw.tensor, right_tensor - left_tensor)
    assert torch.equal(absolute.tensor, (right_tensor - left_tensor).abs())
    assert torch.equal(l2.tensor, raw.tensor) and torch.equal(cosine.tensor, raw.tensor)
    assert raw_heatmap.values.tolist() == [[1.0, 1.0]]
    assert absolute_heatmap.values.tolist() == [[1.0, 1.0]]
    assert l2_heatmap.values.flatten().tolist() == pytest.approx([2.0, 2.0])
    assert cosine_heatmap.values[0, 0].item() == pytest.approx(0)
    assert cosine_heatmap.values[0, 1].item() == pytest.approx(2)


def test_feature_store_refcounts_cleanup_and_capacity():
    store = FeatureStore(max_entries=2, max_bytes=1000)
    artifact = FeatureArtifact(torch.zeros(4, 5, 3), metadata(), {})
    first = store.put(artifact, "run", refcount=2)
    assert store.get(first) is artifact
    store.release(first)
    assert store.stats()["entries"] == 1
    store.release(first)
    assert store.stats()["entries"] == 0
    second = store.put(artifact, "run", refcount=1)
    store.release_run("run")
    with pytest.raises(ValueError, match="feature_not_found"):
        store.get(second)
    tiny = FeatureStore(max_entries=1, max_bytes=100)
    with pytest.raises(MemoryError, match="feature_store_capacity_exceeded"):
        tiny.put(artifact, "run")


def test_dino_reconstructs_non_square_spatial_grid_without_copying_to_cpu():
    adapter = DinoV3Adapter()
    network = SimpleNamespace(config=SimpleNamespace(patch_size=16))
    key = ModelKey("dinov3", "official", "vit-b", "weight", "cpu", "fp32")
    model = LoadedModel((object(), network, "facebook/test"), key, Path("test.pth"), checkpoint_sha256="sha")
    # 2 special tokens followed by a 2x3 patch grid.
    tokens = torch.arange(8 * 4, dtype=torch.float32).reshape(1, 8, 4)
    artifact = adapter.postprocess(model, InferenceRequest(), (tokens, (32, 48), (101, 151)))
    assert artifact.metadata.shape == (2, 3, 4)
    assert artifact.tensor.device == tokens.device
    assert torch.equal(artifact.tensor[0, 0], tokens[0, 2])
    assert artifact.metadata.space.checkpoint_sha256 == "sha"
