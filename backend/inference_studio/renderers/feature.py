from __future__ import annotations

import cv2
import numpy as np

from inference_studio.results import FeatureResult


def render_feature(result: FeatureResult) -> np.ndarray:
    values = np.asarray(result.values, dtype=np.float32)
    if values.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    tokens = result.shape[-2] if len(result.shape) >= 3 else values.size
    side = max(1, int(np.sqrt(max(1, tokens - 1))))
    channels = result.shape[-1] if result.shape else 1
    usable = min(values.size, side * side * channels)
    if usable >= side * side:
        heat = values[:usable].reshape(-1, channels).mean(axis=1)[: side * side].reshape(side, side)
    else:
        heat = np.resize(values, (side, side))
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(heat, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_TURBO)

