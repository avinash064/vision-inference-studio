from __future__ import annotations

from .base import ModelAdapter
from .dinov3 import DinoV3Adapter
from .grounding_dino import GroundingDinoAdapter
from .onnx import OnnxAdapter
from .paddle_ocr import PaddleOcrAdapter
from .rfdetr import RfDetrAdapter
from .rtdetr import RtDetrAdapter
from .rtdetrv2 import RtDetrV2Adapter
from .sam import SamAdapter
from .tesseract import TesseractAdapter
from .ultralytics_sam import Sam2Adapter, Sam3Adapter
from .yolo import YoloAdapter
from .vllm import VllmAdapter


class AdapterRegistry:
    def __init__(self, adapters: list[ModelAdapter] | None = None) -> None:
        instances = adapters or [YoloAdapter(), RtDetrAdapter(), RtDetrV2Adapter(), RfDetrAdapter(), DinoV3Adapter(), GroundingDinoAdapter(), SamAdapter(), Sam2Adapter(), Sam3Adapter(), PaddleOcrAdapter(), TesseractAdapter(), VllmAdapter(), OnnxAdapter()]
        self._adapters = {adapter.name: adapter for adapter in instances}

    def get(self, name: str) -> ModelAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise ValueError(f"Unknown model adapter: {name}") from exc
