"""Fixed-threshold channel-activation XOR for RF-DETR feature levels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def reference_channel_thresholds(
    reference_features: Sequence[torch.Tensor],
    percentile: float = 90.0,
) -> list[torch.Tensor]:
    """Freeze one spatial activation threshold per reference feature channel."""
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between zero and 100")
    quantile = float(percentile) / 100.0
    thresholds: list[torch.Tensor] = []
    with torch.inference_mode():
        for level in reference_features:
            if level.ndim != 4 or level.shape[0] != 1:
                raise ValueError("expected single-image BCHW RF-DETR features")
            flat = level.float().flatten(start_dim=2)
            threshold = torch.quantile(
                flat, quantile, dim=2, keepdim=True
            ).unsqueeze(-1)
            thresholds.append(threshold.detach())
    return thresholds


def fixed_threshold_feature_xor_maps(
    current_features: Sequence[torch.Tensor],
    reference_features: Sequence[torch.Tensor],
    thresholds: Sequence[torch.Tensor],
) -> list[np.ndarray]:
    """Return each level's fraction of channels whose activation state flips.

    The same threshold, frozen from the reference tensor, is applied to both
    current and reference features. No current-frame percentile, min/max, or
    other live normalization is used.
    """
    if not (
        len(current_features) == len(reference_features) == len(thresholds)
    ):
        raise ValueError("RF-DETR XOR level counts differ")
    maps: list[np.ndarray] = []
    with torch.inference_mode():
        for current, reference, threshold in zip(
            current_features, reference_features, thresholds
        ):
            if current.shape != reference.shape:
                raise ValueError("RF-DETR XOR current/reference shapes differ")
            expected = (reference.shape[0], reference.shape[1], 1, 1)
            if tuple(threshold.shape) != expected:
                raise ValueError("RF-DETR XOR channel-threshold shape differs")
            current_binary = current.float() > threshold
            reference_binary = reference.float() > threshold
            value = torch.logical_xor(
                current_binary, reference_binary
            ).float().mean(dim=1)
            maps.append(value[0].cpu().numpy().astype(np.float32))
    return maps


def threshold_summaries(
    thresholds: Sequence[torch.Tensor],
) -> list[dict[str, float | int]]:
    """Compact audit metadata for fixed per-channel thresholds."""
    summaries: list[dict[str, float | int]] = []
    for threshold in thresholds:
        values = threshold.detach().float().cpu().numpy().ravel()
        summaries.append({
            "channels": int(values.size),
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        })
    return summaries


def shared_reference_abs_xor_maps(
    current_features: Sequence[torch.Tensor],
    reference_features: Sequence[torch.Tensor],
    thresholds: Sequence[torch.Tensor],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute ABS and XOR from the exact same frozen reference tensors."""
    if not (
        len(current_features) == len(reference_features) == len(thresholds)
    ):
        raise ValueError("shared ABS/XOR level counts differ")
    absolute: list[np.ndarray] = []
    with torch.inference_mode():
        for current, reference in zip(current_features, reference_features):
            if current.shape != reference.shape:
                raise ValueError("shared ABS/XOR feature shapes differ")
            value = torch.mean(
                torch.abs(current.float() - reference.float()), dim=1
            )
            absolute.append(value[0].cpu().numpy().astype(np.float32))
    xor = fixed_threshold_feature_xor_maps(
        current_features, reference_features, thresholds
    )
    return absolute, xor
