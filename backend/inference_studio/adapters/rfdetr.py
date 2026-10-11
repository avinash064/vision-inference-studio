from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from inference_studio.results import Detection, DetectionResult

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey


class RfDetrAdapter(ModelAdapter):
    name = "rfdetr"
    _classes = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase", "large": "RFDETRLarge"}

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError("RF-DETR runtime is not installed; install requirements-runtimes.txt") from exc
        class_name = self._classes[key.size]
        model_class = getattr(rfdetr, class_name, None)
        if model_class is None:
            raise RuntimeError(f"Installed rfdetr does not expose {class_name}")
        # Supported RF-DETR releases accept model config through **kwargs. Keeping
        # this variation here prevents version-specific branches in the API/engine.
        try:
            model = model_class(pretrain_weights=str(weight_path), device=key.device, amp=key.precision == "fp16")
        except TypeError as exc:
            raise RuntimeError("Installed RF-DETR version does not accept local pretrain_weights/device configuration") from exc
        model.vis_coco_labels = self.coco_label_table(model)
        return LoadedModel(handle=model, key=key, weight_path=weight_path)

    @staticmethod
    def coco_label_table(model) -> dict[int, str] | None:
        """COCO checkpoints predict original COCO category ids (1 = person, 90 classes), but rfdetr 1.6 names
        them from its contiguous 80-entry list, shifting every label (bus -> "train", person -> "bicycle")."""
        names = getattr(model, "class_names", None)
        num_classes = getattr(getattr(model, "model_config", None), "num_classes", None)
        if not (isinstance(names, (list, tuple)) and len(names) == 80 and num_classes in (90, 91)):
            return None
        try:
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ImportError:
            from rfdetr.util.coco_classes import COCO_CLASSES
        return {int(key): str(value) for key, value in dict(COCO_CLASSES).items()}

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor: np.ndarray):
        return model.handle.predict(tensor, threshold=request.confidence, include_source_image=False)

    def postprocess(self, model: LoadedModel, request: InferenceRequest, prediction) -> DetectionResult:
        xyxy = getattr(prediction, "xyxy", [])
        class_ids = getattr(prediction, "class_id", [])
        confidences = getattr(prediction, "confidence", [])
        prediction_data = getattr(prediction, "data", {}) or {}
        class_names = prediction_data.get("class_name", [])
        fallback_names = getattr(model.handle, "class_names", {})
        coco_labels = getattr(model.handle, "vis_coco_labels", None)
        detections = []
        for index, (box, class_id, confidence) in enumerate(zip(xyxy, class_ids, confidences)):
            if coco_labels:
                label = coco_labels.get(int(class_id), class_id)
            else:
                label = class_names[index] if index < len(class_names) else fallback_names.get(int(class_id), class_id) if isinstance(fallback_names, dict) else class_id
            detections.append(Detection(bbox=[float(v) for v in box], class_id=int(class_id), label=str(label or class_id), confidence=float(confidence)))
        return DetectionResult(detections=detections)
