from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from inference_studio.runtime.artifacts import FeatureArtifact


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("Feature visualization requires PyTorch") from exc


def feature_visual(artifact: FeatureArtifact, mode: str = "channel_norm", channel: int = 0) -> Any:
    torch = _torch()
    tensor = artifact.tensor.detach().float()
    if mode in {"channel", "selected_channel"}:
        if channel < 0 or channel >= tensor.shape[-1]:
            raise ValueError(f"Feature channel {channel} is outside 0..{tensor.shape[-1] - 1}")
        return tensor[..., channel]
    if mode in {"pca", "pca_rgb"}:
        flat = tensor.reshape(-1, tensor.shape[-1])
        centered = flat - flat.mean(dim=0, keepdim=True)
        q = min(3, centered.shape[0], centered.shape[1])
        _, _, vectors = torch.pca_lowrank(centered, q=q)
        projected = centered @ vectors[:, :q]
        if q < 3:
            projected = torch.nn.functional.pad(projected, (0, 3 - q))
        return projected.reshape(*tensor.shape[:2], 3)
    return torch.linalg.vector_norm(tensor, dim=-1)


def render_heatmap(values: Any, width: int, height: int) -> np.ndarray:
    array = values.detach().float().cpu().numpy() if hasattr(values, "detach") else np.asarray(values, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 3:
        channels = []
        for index in range(3):
            band = array[..., index]
            low, high = float(np.min(band)), float(np.max(band))
            channels.append(np.zeros_like(band, dtype=np.uint8) if high <= low else ((band - low) * 255.0 / (high - low)).astype(np.uint8))
        rgb = np.stack(channels, axis=-1)
        return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (width, height), interpolation=cv2.INTER_CUBIC)
    low, high = float(np.min(array)), float(np.max(array))
    scaled = np.zeros_like(array, dtype=np.uint8) if high <= low else ((array - low) * 255.0 / (high - low)).astype(np.uint8)
    resized = cv2.resize(scaled, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(resized, cv2.COLORMAP_TURBO)
