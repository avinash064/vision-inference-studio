from __future__ import annotations

from pathlib import Path
import cv2
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
        # Supported RF-DETR releases accept model config through **kwargs. Keeping
        # this variation here prevents version-specific branches in the API/engine.
        try:
            model = model_class(pretrain_weights=str(weight_path), device=config.device, amp=config.precision == "fp16")
        except TypeError as exc:
            raise RuntimeError("Installed RF-DETR version does not accept local pretrain_weights/device configuration") from exc
        return LoadedModel(handle=model, config=config, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, input_data: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

    def infer(self, model: LoadedModel, tensor: np.ndarray):
        return model.handle.predict(tensor, threshold=model.config.confidence, include_source_image=False)

    def postprocess(self, model: LoadedModel, prediction) -> DetectionResult:
        xyxy = getattr(prediction, "xyxy", [])
        class_ids = getattr(prediction, "class_id", [])
        confidences = getattr(prediction, "confidence", [])
        prediction_data = getattr(prediction, "data", {}) or {}
        class_names = prediction_data.get("class_name", [])
        fallback_names = getattr(model.handle, "class_names", {})
        detections = []
        for index, (box, class_id, confidence) in enumerate(zip(xyxy, class_ids, confidences)):
            label = class_names[index] if index < len(class_names) else fallback_names.get(int(class_id), class_id) if isinstance(fallback_names, dict) else class_id
            detections.append(Detection(bbox=[float(v) for v in box], class_id=int(class_id), label=str(label or class_id), confidence=float(confidence)))
        return DetectionResult(detections=detections)
