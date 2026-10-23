from __future__ import annotations

from pathlib import Path

from .base import LoadedModel, ModelKey
from .yolo import YoloAdapter


class RtDetrAdapter(YoloAdapter):
    """Ultralytics RT-DETR checkpoints using the standard detection result contract."""

    name = "rtdetr"

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            from ultralytics import RTDETR
        except ImportError as exc:
            raise RuntimeError("RT-DETR runtime is not installed; install requirements-runtimes.txt") from exc
        model = RTDETR(str(weight_path))
        return LoadedModel(handle=model, key=key, weight_path=weight_path)
