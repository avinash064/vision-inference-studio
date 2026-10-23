from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_studio.adapters import AdapterRegistry, InferenceRequest, LoadedModel, ModelConfig
from inference_studio.adapters.rfdetr import RfDetrAdapter
from inference_studio.adapters.rtdetr import RtDetrAdapter
from inference_studio.adapters.rtdetrv2 import RtDetrV2Adapter, RtDetrV2Handle
from inference_studio.adapters.paddle_ocr import PaddleOcrAdapter
from inference_studio.adapters.tesseract import TesseractAdapter
from inference_studio.adapters.vllm import VllmAdapter
from inference_studio.adapters.grounding_dino import GroundingDinoAdapter
from inference_studio.adapters.sam import SamAdapter, SamHandle
from inference_studio.adapters.ultralytics_sam import Sam2Adapter, Sam3Adapter, to_segmentation
from inference_studio.adapters.yolo import YoloAdapter
from inference_studio.results import DetectionResult


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def numpy(self):
        return np.asarray(self.values)


def loaded(family: str = "rfdetr") -> LoadedModel:
    version = "v1" if family == "rfdetr" else "ultralytics" if family == "rtdetr" else "v11"
    size = "large" if family == "rfdetr" else "custom" if family == "rtdetr" else "n"
    config = ModelConfig(family=family, version=version, size=size, weight_id="test")
    return LoadedModel(handle=SimpleNamespace(class_names={4: "fallback"}), key=config.model_key(), weight_path=Path("test.pth" if family == "rfdetr" else "test.pt"))


def test_adapter_registry_resolves_model_specific_code_once():
    registry = AdapterRegistry()
    assert registry.get("rfdetr").name == "rfdetr"
    assert registry.get("rtdetr").name == "rtdetr"
    assert registry.get("rtdetrv2").name == "rtdetrv2"
    assert registry.get("yolo").name == "yolo"
    assert registry.get("grounding-dino").name == "grounding-dino"
    assert registry.get("sam").name == "sam"
    assert registry.get("sam2").name == "sam2"
    assert registry.get("sam3").name == "sam3"
    assert registry.get("paddleocr").name == "paddleocr"
    assert registry.get("tesseract").name == "tesseract"
    assert registry.get("vllm").name == "vllm"


def test_rfdetr_normalizes_supervision_detection_and_rgb_preprocessing():
    adapter = RfDetrAdapter()
    prediction = SimpleNamespace(xyxy=np.asarray([[10, 20, 80, 90]], dtype=float), class_id=np.asarray([4]), confidence=np.asarray([.87]), data={"class_name": np.asarray(["skip_car"])})
    result = adapter.postprocess(loaded(), InferenceRequest(), prediction)
    assert result == DetectionResult.model_validate({"result_type": "detections", "detections": [{"bbox": [10, 20, 80, 90], "class_id": 4, "label": "skip_car", "confidence": .87}]})
    bgr = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    assert adapter.preprocess(loaded(), InferenceRequest(), bgr).tolist() == [[[3, 2, 1]]]


def test_rfdetr_coco_checkpoints_use_coco_category_ids():
    adapter = RfDetrAdapter()
    handle = SimpleNamespace(class_names=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train"] + ["x"] * 73, model_config=SimpleNamespace(num_classes=90))
    handle.vis_coco_labels = adapter.coco_label_table(handle)
    prediction = SimpleNamespace(xyxy=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float), class_id=np.asarray([6, 1]), confidence=np.asarray([.97, .9]), data={"class_name": np.asarray(["train", "bicycle"])})
    model = loaded()
    model.handle = handle
    assert [d.label for d in adapter.postprocess(model, InferenceRequest(), prediction).detections] == ["bus", "person"]
    # Fine-tuned checkpoints keep their own names.
    assert adapter.coco_label_table(SimpleNamespace(class_names=["skip_car"], model_config=SimpleNamespace(num_classes=1))) is None


def test_rfdetr_patch_size_selects_matching_large_architecture(monkeypatch):
    import rfdetr

    calls = []

    class CurrentLarge:
        class_names = {}
        model_config = SimpleNamespace(num_classes=13)
        def __init__(self, **kwargs): calls.append(("current", kwargs))

    class LegacyLarge:
        class_names = {}
        model_config = SimpleNamespace(num_classes=13)
        def __init__(self, **kwargs): calls.append(("legacy", kwargs))

    monkeypatch.setattr(rfdetr, "RFDETRLarge", CurrentLarge)
    monkeypatch.setattr(rfdetr, "RFDETRLargeDeprecated", LegacyLarge)
    adapter = RfDetrAdapter()
    legacy = ModelConfig(family="rfdetr", version="v1", size="large", patch_size=14, weight_id="legacy")
    current = ModelConfig(family="rfdetr", version="v1", size="large", patch_size=16, weight_id="current")
    assert adapter.load(legacy.model_key(), Path("legacy.pth")).handle.vis_runtime_class == "RFDETRLargeDeprecated"
    assert adapter.load(current.model_key(), Path("current.pth")).handle.vis_runtime_class == "RFDETRLarge"
    assert calls == [
        ("legacy", {"pretrain_weights": "legacy.pth", "device": "cpu", "amp": False, "patch_size": 14}),
        ("current", {"pretrain_weights": "current.pth", "device": "cpu", "amp": False, "patch_size": 16}),
    ]
    with pytest.raises(RuntimeError, match="does not support patch size 14"):
        adapter.runtime_class("medium", 14)


def test_yolo_normalizes_ultralytics_detection():
    adapter = YoloAdapter()
    output = SimpleNamespace(names={2: "skip_car"}, boxes=SimpleNamespace(xyxy=FakeTensor([[1, 2, 30, 40]]), cls=FakeTensor([2]), conf=FakeTensor([.93])))
    result = adapter.postprocess(loaded("yolo"), InferenceRequest(), output)
    assert result.result_type == "detections"
    assert result.detections[0].model_dump() == {"bbox": [1.0, 2.0, 30.0, 40.0], "class_id": 2, "label": "skip_car", "confidence": .93}


def test_rtdetr_uses_ultralytics_runtime_and_detection_contract(monkeypatch):
    import ultralytics

    output = SimpleNamespace(names={1: "vehicle"}, boxes=SimpleNamespace(xyxy=FakeTensor([[4, 5, 24, 35]]), cls=FakeTensor([1]), conf=FakeTensor([.89])))
    calls = {}

    class FakeRtDetr:
        def __init__(self, path): calls["path"] = path
        def predict(self, **kwargs): calls.update(kwargs); return [output]

    monkeypatch.setattr(ultralytics, "RTDETR", FakeRtDetr)
    config = ModelConfig(family="rtdetr", version="ultralytics", size="custom", weight_id="test", device="cuda:0", precision="fp16", confidence=.4)
    adapter = RtDetrAdapter(); model = adapter.load(config.model_key(), Path("custom-rtdetr.pt"))
    raw = adapter.infer(model, config.inference_request(), np.zeros((32, 32, 3), np.uint8))
    result = adapter.postprocess(model, config.inference_request(), raw)
    assert calls["path"] == "custom-rtdetr.pt"
    assert calls["conf"] == .4 and calls["device"] == "cuda:0" and calls["half"] is True
    assert result.detections[0].model_dump() == {"bbox": [4.0, 5.0, 24.0, 35.0], "class_id": 1, "label": "vehicle", "confidence": .89}


def test_rtdetrv2_uses_bmd_category_ids_and_detection_contract():
    import torch

    config = ModelConfig(family="rtdetrv2", version="bmd-45", size="r101-x", weight_id="test", confidence=.5)
    model = LoadedModel(
        handle=RtDetrV2Handle(network=object(), postprocessor=object(), class_names={1: "Hatchback", 13: "Van"}),
        key=config.model_key(),
        weight_path=Path("best.pth"),
    )
    output = (
        torch.tensor([[1, 13, 4]]),
        torch.tensor([[[1, 2, 30, 40], [5, 6, 50, 60], [0, 0, 2, 2]]], dtype=torch.float32),
        torch.tensor([[.91, .72, .2]]),
    )
    result = RtDetrV2Adapter().postprocess(model, config.inference_request(), output)
    assert [item.model_dump() for item in result.detections] == [
        {"bbox": [1.0, 2.0, 30.0, 40.0], "class_id": 1, "label": "Hatchback", "confidence": pytest.approx(.91)},
        {"bbox": [5.0, 6.0, 50.0, 60.0], "class_id": 13, "label": "Van", "confidence": pytest.approx(.72)},
    ]


def test_rtdetrv2_identifies_native_checkpoint_classifier():
    import torch

    state = {"decoder.enc_score_head.weight": torch.zeros(14, 256)}
    assert RtDetrV2Adapter._num_classes(state) == 14
    assert RtDetrV2Adapter._state_dict({"ema": {"module": state}}) is state


def test_paddleocr_normalizes_v3_result_boxes_and_scores():
    config = ModelConfig(family="paddleocr", version="PP-OCRv5", size="mobile", ocr_language="en", confidence=.5)
    model = LoadedModel(handle=object(), key=config.model_key(), weight_path=Path())
    output = [SimpleNamespace(json={"res": {"rec_texts": ["Skip 42", "low"], "rec_scores": [.94, .2], "rec_polys": [np.asarray([[1, 2], [20, 2], [20, 10], [1, 10]]), np.zeros((4, 2))]}})]
    result = PaddleOcrAdapter().postprocess(model, config.inference_request(), output)
    assert result.text == "Skip 42" and result.language == "en"
    assert result.spans[0].bbox == [1.0, 2.0, 20.0, 10.0]
    assert result.spans[0].confidence == .94


def test_tesseract_normalizes_word_data():
    config = ModelConfig(family="tesseract", version="5", size="system", ocr_language="en", confidence=.5)
    model = LoadedModel(handle=object(), key=config.model_key(), weight_path=Path())
    output = {"text": ["Skip", "", "Car"], "conf": ["96", "-1", "81"], "left": [2, 0, 30], "top": [3, 0, 3], "width": [20, 0, 18], "height": [10, 0, 10]}
    result = TesseractAdapter().postprocess(model, config.inference_request(), output)
    assert result.text == "Skip Car"
    assert result.spans[1].bbox == [30.0, 3.0, 48.0, 13.0]


def test_vllm_normalizes_openai_multimodal_content():
    config = ModelConfig(family="vllm", version="openai-compatible", size="remote", endpoint_url="http://localhost:8000", endpoint_model="vision", prompt_mode="text", text_prompt="Read all text")
    model = VllmAdapter().load(config.model_key(), Path())
    assert model.handle["endpoint"] == "http://localhost:8000/v1/chat/completions"
    result = VllmAdapter().postprocess(model, config.inference_request(), {"choices": [{"message": {"content": [{"type": "text", "text": "SKIP CAR 12"}]}}]})
    assert result.text == "SKIP CAR 12" and result.spans[0].bbox is None


def test_grounding_dino_builds_official_caption_and_pixel_boxes():
    adapter = GroundingDinoAdapter()
    assert adapter.normalize_prompt("Car, truck; PERSON") == "car . truck . person ."
    # Official outputs are normalized cx, cy, w, h; boxes are clipped to the image.
    detections = adapter.to_detections(np.asarray([[.5, .5, .2, .4], [.95, .5, .2, .2]]), np.asarray([.88, .4]), ["skip car", ""], height=40, width=50)
    assert detections[0].model_dump() == {"bbox": [20.0, 12.0, 30.0, 28.0], "class_id": 0, "label": "skip car", "confidence": .88}
    assert detections[1].bbox == [42.5, 16.0, 50.0, 24.0]
    assert detections[1].label == "prompt"


def fake_ultralytics_result(names=None):
    mask = np.zeros((2, 32, 32), dtype=bool)
    mask[0, 5:20, 7:24] = True
    mask[1, 25:30, 2:6] = True
    boxes = SimpleNamespace(conf=FakeTensor([.81, .42]), cls=FakeTensor([1, 0]))
    return SimpleNamespace(masks=SimpleNamespace(data=FakeTensor(mask)), boxes=boxes, names=names or {0: "0", 1: "1"})


def test_ultralytics_sam_results_convert_to_segments():
    point = ModelConfig(family="sam2", version="2.1", size="base-plus", weight_id="test", prompt_mode="point", point_prompt="12,10")
    result = to_segmentation(fake_ultralytics_result(), point.inference_request())
    assert result.prompt == "12,10"
    assert [mask.label for mask in result.masks] == ["segment 1", "segment 2"]
    assert result.masks[0].bbox == [7.0, 5.0, 23.0, 19.0] and result.masks[0].area == 255 and result.masks[0].polygons
    text = ModelConfig(family="sam3", version="3.0", size="base", weight_id="test", prompt_mode="text", text_prompt="wheel, rail")
    semantic = to_segmentation(fake_ultralytics_result(["wheel", "rail"]), text.inference_request(), semantic=True)  # SAM 3 reports names as a list
    assert [(mask.label, mask.class_id, mask.confidence) for mask in semantic.masks] == [("rail", 1, .81), ("wheel", 0, .42)]


def test_sam2_uses_canonical_checkpoint_names_and_spatial_prompts():
    assert Sam2Adapter.checkpoint_name("2.1", "base-plus") == "sam2.1_b.pt"
    assert Sam2Adapter.checkpoint_name("2.0", "tiny") == "sam2_t.pt"
    box = ModelConfig(family="sam2", version="2.1", size="large", weight_id="test", prompt_mode="box", box_prompt="1,2,20,30")
    model = LoadedModel(handle=object(), key=box.model_key(), weight_path=Path("test.pt"))
    assert Sam2Adapter().preprocess(model, box.inference_request(), np.zeros((32, 32, 3), np.uint8))["prompts"] == {"bboxes": [[1.0, 2.0, 20.0, 30.0]]}
    outside = box.model_copy(update={"box_prompt": "1,2,40,30"})
    with pytest.raises(ValueError):
        Sam2Adapter().preprocess(LoadedModel(handle=object(), key=outside.model_key(), weight_path=Path("test.pt")), outside.inference_request(), np.zeros((32, 32, 3), np.uint8))


def test_sam3_routes_text_to_concepts_and_points_to_instances():
    text = ModelConfig(family="sam3", version="3.0", size="base", weight_id="test", prompt_mode="text", text_prompt="wheel; rail, ")
    prepared = Sam3Adapter().preprocess(LoadedModel(handle=object(), key=text.model_key(), weight_path=Path("sam3.pt")), text.inference_request(), np.zeros((8, 8, 3), np.uint8))
    assert prepared["semantic"] and prepared["prompts"] == {"text": ["wheel", "rail"]}
    point = text.model_copy(update={"prompt_mode": "point", "point_prompt": "3,4", "point_foreground": False})
    prepared = Sam3Adapter().preprocess(LoadedModel(handle=object(), key=point.model_key(), weight_path=Path("sam3.pt")), point.inference_request(), np.zeros((8, 8, 3), np.uint8))
    assert not prepared["semantic"] and prepared["prompts"] == {"points": [[3.0, 4.0]], "labels": [0]}


class FakeSamProcessor:
    def post_process_masks(self, *_args, **_kwargs):
        mask = np.zeros((1, 1, 32, 32), dtype=bool)
        mask[0, 0, 5:20, 7:24] = True
        return [mask]


def test_sam_converts_best_mask_to_json_safe_polygons():
    adapter = SamAdapter()
    config = ModelConfig(family="sam", version="v1", size="vit-b", weight_id="test", prompt_mode="point", point_prompt="12,10")
    model = LoadedModel(handle=SamHandle(FakeSamProcessor(), object()), key=config.model_key(), weight_path=Path("test.safetensors"))
    raw = SimpleNamespace(pred_masks=np.empty(0), iou_scores=np.asarray([[[.91]]]))
    result = adapter.postprocess(model, config.inference_request(), (raw, {"original_sizes": [[32, 32]], "reshaped_input_sizes": [[32, 32]]}))
    assert result.result_type == "segmentation"
    assert result.prompt == "12,10"
    assert result.masks[0].area == 255
    assert result.masks[0].bbox == [7.0, 5.0, 23.0, 19.0]
    assert result.masks[0].polygons
