from __future__ import annotations

import cv2
import numpy as np

from inference_studio.results import TensorResult


def render_tensor(result: TensorResult) -> np.ndarray:
    canvas = np.full((360, 640, 3), 20, dtype=np.uint8)
    lines = [f"shape: {result.shape}", f"dtype: {result.dtype}", "values:"]
    lines.extend("  " + ", ".join(f"{value:.4g}" for value in result.values[index:index + 8]) for index in range(0, min(40, len(result.values)), 8))
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (24, 38 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 228, 240), 1, cv2.LINE_AA)
    return canvas

