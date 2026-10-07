from __future__ import annotations

import cv2
import numpy as np

from inference_studio.results import DetectionResult


def render_detections(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    canvas = image.copy()
    for item in result.detections:
        x1, y1, x2, y2 = (int(value) for value in item.bbox)
        color = (67, 207, 142)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{item.label} {item.confidence:.2f}"
        cv2.putText(canvas, text, (x1, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return canvas

