from __future__ import annotations

from .base import ModelAdapter
from .dinov3 import DinoV3Adapter
from .onnx import OnnxAdapter
from .rfdetr import RfDetrAdapter
from .yolo import YoloAdapter


class AdapterRegistry:
    def __init__(self, adapters: list[ModelAdapter] | None = None) -> None:
        instances = adapters or [YoloAdapter(), RfDetrAdapter(), DinoV3Adapter(), OnnxAdapter()]
        self._adapters = {adapter.name: adapter for adapter in instances}

    def get(self, name: str) -> ModelAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise ValueError(f"Unknown model adapter: {name}") from exc

