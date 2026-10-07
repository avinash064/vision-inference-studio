from __future__ import annotations

import inspect
from pathlib import Path
import numpy as np

from inference_studio.results import Detection, DetectionResult

from .base import LoadedModel, ModelAdapter, ModelConfig


class RfDetrAdapter(ModelAdapter):
    name = "rfdetr"
    _classes = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase", "large": "RFDETRLarge"}

    def load(self, config: ModelConfig, weight_path: Path) -> LoadedModel:
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError("RF-DETR runtime is not installed; install requirements-runtimes.txt") from exc
        class_name = self._classes[config.size]
        model_class = getattr(rfdetr, class_name, None)
        if model_class is None:
            raise RuntimeError(f"Installed rfdetr does not expose {class_name}")
        parameters = inspect.signature(model_class).parameters
        kwargs = {}
        for name in ("pretrain_weights", "checkpoint", "weights"):
            if name in parameters:
                kwargs[name] = str(weight_path)
                break
        if not kwargs:
            raise RuntimeError("Installed RF-DETR version has no supported local-weight parameter")
        model = model_class(**kwargs)
        return LoadedModel(handle=model, config=config, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, input_data: np.ndarray) -> np.ndarray:
        return input_data

    def infer(self, model: LoadedModel, tensor: np.ndarray):
        return model.handle.predict(tensor, threshold=model.config.confidence)

    def postprocess(self, model: LoadedModel, prediction) -> DetectionResult:
        xyxy = getattr(prediction, "xyxy", [])
        class_ids = getattr(prediction, "class_id", [])
        confidences = getattr(prediction, "confidence", [])
        names = getattr(model.handle, "class_names", {})
        detections = [Detection(bbox=[float(v) for v in box], class_id=int(class_id), label=str(names.get(int(class_id), class_id) if isinstance(names, dict) else class_id), confidence=float(confidence)) for box, class_id, confidence in zip(xyxy, class_ids, confidences)]
        return DetectionResult(detections=detections)
