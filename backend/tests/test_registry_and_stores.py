from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from inference_studio.assets import AssetError
from inference_studio.models.registry import ModelRegistryError
from inference_studio.models.weights import WeightError
from inference_studio.renderers import RendererRegistry
from inference_studio.results import FeatureResult, TensorResult
from inference_studio.nodes import NodeRegistry


def test_model_registry_contains_declared_families(model_registry):
    assert set(model_registry.manifest.families) == {"yolo", "rfdetr", "dinov3", "onnx"}
    assert model_registry.validate_selection("yolo", "v11", "n").adapter == "yolo"
    with pytest.raises(ModelRegistryError):
        model_registry.validate_selection("dinov3", "official", "not-a-size")


def test_node_registry_exposes_typed_server_definitions():
    registry = NodeRegistry()
    assert {item.type for item in registry.list()} >= {"image_source", "video_source", "model", "detection_preview"}
    model = registry.get("model")
    assert model.inputs[0].data_types == ["image", "video"]
    assert "detections" in model.outputs[0].data_types


def test_weight_store_uses_internal_ids_and_family_directories(stores):
    weights, _ = stores
    record = weights.save(BytesIO(b"safe bytes"), "../../outside.pt", "yolo")
    assert record.weight_id.startswith("wgt_")
    assert Path(record.path).parent.name == "yolo"
    assert Path(record.path).is_file()
    with pytest.raises(WeightError):
        weights.save(BytesIO(b"x"), "model.onnx", "yolo")
    with pytest.raises(WeightError):
        weights.get("../../etc/passwd")


def test_asset_store_reads_metadata_and_rejects_bad_content(stores, image_bytes):
    _, assets = stores
    record = assets.save(BytesIO(image_bytes), "frame.png", "image")
    assert (record.width, record.height) == (64, 48)
    assert assets.get(record.asset_id).sha256 == record.sha256
    with pytest.raises(AssetError):
        assets.save(BytesIO(b"not an image"), "broken.png", "image")


def test_renderer_registry_selects_standard_result_renderer():
    registry = RendererRegistry()
    feature = registry.render(None, FeatureResult(shape=[1, 4, 2], values=list(range(8))))
    tensor = registry.render(None, TensorResult(shape=[2], dtype="float32", values=[1, 2]))
    assert feature.shape == (512, 512, 3)
    assert tensor.shape == (360, 640, 3)
