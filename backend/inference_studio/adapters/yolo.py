from __future__ import annotations

from pathlib import Path
import numpy as np

from inference_studio.results import Detection, DetectionResult

from .base import LoadedModel, ModelAdapter, ModelConfig


class YoloAdapter(ModelAdapter):
    name = "yolo"

    def load(self, config: ModelConfig, weight_path: Path) -> LoadedModel:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO runtime is not installed; install requirements-runtimes.txt") from exc
        model = YOLO(str(weight_path), task="detect")
        return LoadedModel(handle=model, config=config, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, input_data: np.ndarray) -> np.ndarray:
        return input_data

    def infer(self, model: LoadedModel, tensor: np.ndarray):
        return model.handle.predict(source=tensor, conf=model.config.confidence, device=model.config.device, half=model.config.precision == "fp16", verbose=False)[0]

    def postprocess(self, model: LoadedModel, output) -> DetectionResult:
        detections: list[Detection] = []
        names = output.names
        if output.boxes is not None:
            for box, class_id, confidence in zip(output.boxes.xyxy.cpu().tolist(), output.boxes.cls.cpu().tolist(), output.boxes.conf.cpu().tolist()):
                index = int(class_id)
                detections.append(Detection(bbox=box, class_id=index, label=str(names.get(index, index)), confidence=float(confidence)))
        return DetectionResult(detections=detections)
