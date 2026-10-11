"""Compatibility facade for the v0.5 runtime feature import path."""

from inference_studio.features.ops import (
    activation_xor,
    difference,
    heatmap_statistics,
    mahalanobis_difference,
    normalize,
    tensor_summary as summary,
)
from inference_studio.features.visualization import feature_visual as feature_heatmap, render_heatmap

__all__ = [
    "activation_xor", "difference", "feature_heatmap", "heatmap_statistics",
    "mahalanobis_difference", "normalize", "render_heatmap", "summary",
]
