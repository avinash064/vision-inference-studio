from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from inference_studio.assets import AssetError
from inference_studio.models.registry import ModelRegistryError
from inference_studio.models.weights import WeightError
from inference_studio.renderers import RendererRegistry
from inference_studio.results import FeatureResult, OcrResult, OcrSpan, SegmentationMask, SegmentationResult, TensorResult
from inference_studio.nodes import NodeRegistry


def test_model_registry_contains_declared_families(model_registry):
    assert set(model_registry.manifest.families) == {"yolo", "yolo-seg", "rtdetr", "rtdetrv2", "rfdetr", "dinov3", "grounding-dino", "sam", "sam2", "sam3", "paddleocr", "tesseract", "vllm", "onnx"}
    assert set(model_registry.get("yolo").versions) == {"v3", "v5", "v6", "v7-export", "v8", "v9", "v10", "v11", "v12", "v26"}
    assert model_registry.validate_selection("yolo", "v11", "n").adapter == "yolo"
    assert model_registry.validate_selection("yolo-seg", "v11", "n").result_type == "segmentation"
    assert model_registry.validate_selection("rtdetr", "ultralytics", "custom").adapter == "rtdetr"
    assert model_registry.validate_selection("rtdetrv2", "bmd-45", "r101-x").adapter == "rtdetrv2"
    assert model_registry.validate_selection("rfdetr", "v1", "large").patch_sizes["large"] == [14, 16]
    assert model_registry.validate_selection("grounding-dino", "official", "tiny").prompt_modes == ["text"]
    assert model_registry.validate_selection("sam", "v1", "vit-b").prompt_modes == ["point", "box"]
    assert model_registry.validate_selection("sam2", "2.1", "base-plus").adapter == "sam2"
    assert model_registry.validate_selection("sam3", "3.0", "base").prompt_modes == ["text", "point", "box"]
    assert not model_registry.validate_selection("paddleocr", "PP-OCRv5", "mobile").requires_weights
    assert model_registry.validate_selection("vllm", "openai-compatible", "remote").runtime_fields == ["ocr_language", "endpoint_url", "endpoint_model"]
    assert model_registry.validate_selection("grounding-dino", "official", "tiny").controls == ["text_threshold"]
    with pytest.raises(ModelRegistryError):
        model_registry.validate_selection("dinov3", "official", "not-a-size")


def test_node_registry_exposes_typed_server_definitions():
    registry = NodeRegistry()
    assert {item.type for item in registry.list()} >= {"image_source", "video_source", "model", "detection_preview"}
    model = registry.get("model")
    assert model.inputs[0].data_types == ["image", "video"]
    assert {"detections", "segmentation", "ocr"}.issubset(model.outputs[0].data_types)
    assert next(field for field in model.config if field.id == "patch_size").options == [14, 16]
    assert registry.get("segmentation_preview").inputs[0].data_types == ["segmentation"]
    assert registry.get("ocr_preview").inputs[0].data_types == ["ocr"]


def test_weight_store_uses_internal_ids_and_family_directories(stores):
    weights, _ = stores
    record = weights.save(BytesIO(b"safe bytes"), "../../outside.pt", "yolo")
    assert record.weight_id.startswith("wgt_")
    assert Path(record.path).parent.name == "yolo"
    assert Path(record.path).is_file()
    with pytest.raises(WeightError):
        weights.save(BytesIO(b"x"), "model.safetensors", "yolo")
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


def test_renderer_registry_overlays_segmentation_polygons():
    registry = RendererRegistry()
    image = __import__("numpy").zeros((40, 50, 3), dtype="uint8")
    result = SegmentationResult(
        prompt_mode="box",
        prompt="5,5,30,30",
        masks=[SegmentationMask(class_id=0, label="segment 1", confidence=.9, bbox=[5, 5, 30, 30], area=625, polygons=[[5, 5, 30, 5, 30, 30, 5, 30]])],
    )
    rendered = registry.render(image, result)
    assert rendered.shape == image.shape
    assert rendered.sum() > 0


def test_renderer_registry_overlays_ocr_regions():
    registry = RendererRegistry()
    image = __import__("numpy").zeros((40, 80, 3), dtype="uint8")
    rendered = registry.render(image, OcrResult(text="skip car", spans=[OcrSpan(text="skip car", confidence=.92, bbox=[4, 5, 60, 24])]))
    assert rendered.shape == image.shape
    assert rendered.sum() > 0
