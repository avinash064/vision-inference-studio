from __future__ import annotations

import cv2
import numpy as np

from inference_studio.results import OcrResult


def render_ocr(image: np.ndarray, result: OcrResult) -> np.ndarray:
    canvas = image.copy()
    for span in result.spans:
        if span.bbox is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in span.bbox)
        color = (245, 176, 65)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = span.text.replace("\n", " ").strip()
        if len(label) > 48:
            label = label[:45] + "..."
        cv2.putText(canvas, f"{label} {span.confidence:.2f}", (x1, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas
