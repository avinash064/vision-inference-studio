from __future__ import annotations

import cv2
import numpy as np

from inference_studio.results import SegmentationResult


_COLORS = (
    (67, 207, 142),
    (239, 160, 74),
    (190, 109, 255),
    (88, 184, 255),
    (226, 105, 128),
)


def render_segmentation(image: np.ndarray, result: SegmentationResult) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()
    outlines: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    for index, item in enumerate(result.masks):
        color = _COLORS[index % len(_COLORS)]
        for polygon in item.polygons:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) < 3:
                continue
            contour = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [contour], color)
            outlines.append((contour, color))

    canvas = cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0)
    for contour, color in outlines:
        cv2.polylines(canvas, [contour], True, color, 2, cv2.LINE_AA)

    for index, item in enumerate(result.masks):
        color = _COLORS[index % len(_COLORS)]
        x1, y1, x2, y2 = (int(value) for value in item.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        text = f"{item.label} {item.confidence:.2f}"
        cv2.putText(
            canvas,
            text,
            (x1, max(16, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas
