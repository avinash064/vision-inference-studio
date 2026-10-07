from __future__ import annotations

from collections.abc import Callable

import numpy as np

from inference_studio.results import DetectionResult, FeatureResult, InferenceResult, TensorResult

from .detection import render_detections
from .feature import render_feature
from .tensor import render_tensor


class RendererRegistry:
    def __init__(self) -> None:
        self._renderers: dict[str, Callable[[np.ndarray | None, InferenceResult], np.ndarray]] = {
            "detections": lambda image, result: render_detections(self._require_image(image), DetectionResult.model_validate(result)),
            "feature_map": lambda _image, result: render_feature(FeatureResult.model_validate(result)),
            "embedding": lambda _image, result: render_feature(FeatureResult.model_validate(result)),
            "tensor": lambda _image, result: render_tensor(TensorResult.model_validate(result)),
        }

    def render(self, image: np.ndarray | None, result: InferenceResult) -> np.ndarray:
        try:
            renderer = self._renderers[result.result_type]
        except KeyError as exc:
            raise ValueError(f"No renderer for result type: {result.result_type}") from exc
        return renderer(image, result)

    @staticmethod
    def _require_image(image: np.ndarray | None) -> np.ndarray:
        if image is None:
            raise ValueError("Detection rendering requires the source image")
        return image

