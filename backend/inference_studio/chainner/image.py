from __future__ import annotations

import numpy as np


class ChainnerImageAdapter:
    """Explicit image boundary without file or lossy codec round-trips."""

    @staticmethod
    def to_chainner(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.float32:
            return np.clip(image, 0, 1)
        if np.issubdtype(image.dtype, np.integer):
            maximum = np.iinfo(image.dtype).max
            return np.clip(image.astype(np.float32) / maximum, 0, 1)
        return np.clip(image.astype(np.float32), 0, 1)

    @staticmethod
    def to_vision(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        if np.issubdtype(image.dtype, np.floating):
            return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        if np.issubdtype(image.dtype, np.integer):
            maximum = np.iinfo(image.dtype).max
            return np.rint(image.astype(np.float32) / maximum * 255).astype(np.uint8)
        raise TypeError(f"Unsupported image dtype: {image.dtype}")
