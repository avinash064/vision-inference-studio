from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from inference_studio.results import Detection, DetectionResult

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey


class RfDetrAdapter(ModelAdapter):
    name = "rfdetr"
    _classes = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase", "large": "RFDETRLarge"}
    _default_patch_sizes = {"base": 14, "nano": 16, "small": 16, "medium": 16, "large": 16}

    @classmethod
    def runtime_class(cls, size: str, patch_size: int) -> tuple[str, int]:
        """Resolve a size/patch pair to an RF-DETR runtime architecture.

        RF-DETR Large patch 14 is the legacy ViT-B layout; patch 16 is the
        current ViT-S layout. They have different hidden dimensions, decoder
        heads and projector scales, so changing only the patch embedding is
        invalid. A zero patch size preserves old workflows by using the class
        default (and RFDETRLarge's upstream compatibility fallback).
        """
        if size not in cls._classes:
            raise RuntimeError(f"Unknown RF-DETR size: {size}")
        resolved = patch_size or cls._default_patch_sizes[size]
        allowed = {"base": {14}, "nano": {16}, "small": {16}, "medium": {16}, "large": {14, 16}}[size]
        if resolved not in allowed:
            choices = ", ".join(str(value) for value in sorted(allowed))
            raise RuntimeError(f"RF-DETR {size} does not support patch size {resolved}; choose {choices}")
        if size == "large" and resolved == 14:
            return "RFDETRLargeDeprecated", resolved
        return cls._classes[size], resolved

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError("RF-DETR runtime is not installed; install requirements-runtimes.txt") from exc
        class_name, patch_size = self.runtime_class(key.size, key.patch_size)
        model_class = getattr(rfdetr, class_name, None)
        if model_class is None:
            raise RuntimeError(f"Installed rfdetr does not expose {class_name}")
        # Supported RF-DETR releases accept model config through **kwargs. Keeping
        # this variation here prevents version-specific branches in the API/engine.
        try:
            kwargs = {"pretrain_weights": str(weight_path), "device": key.device, "amp": key.precision == "fp16"}
            if key.patch_size:
                kwargs["patch_size"] = patch_size
            model = model_class(**kwargs)
        except TypeError as exc:
            raise RuntimeError("Installed RF-DETR version does not accept local pretrain_weights/device configuration") from exc
        model.vis_coco_labels = self.coco_label_table(model)
        model.vis_patch_size = patch_size
        model.vis_runtime_class = class_name
        return LoadedModel(handle=model, key=key, weight_path=weight_path)

    def metadata(self, model: LoadedModel) -> dict:
        value = super().metadata(model)
        value.update(
            patch_size=getattr(model.handle, "vis_patch_size", model.key.patch_size or None),
            runtime_class=getattr(model.handle, "vis_runtime_class", type(model.handle).__name__),
        )
        return value

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
