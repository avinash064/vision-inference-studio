from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_studio.runtime.artifacts import FeatureArtifact
from inference_studio.runtime.values import HeatmapValue


class FeatureSpaceMismatch(ValueError):
    pass


class MahGeometryMismatch(ValueError):
    pass


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("Feature operations require PyTorch") from exc


def tensor_summary(tensor: Any) -> dict[str, float]:
    values = tensor.detach().float()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def heatmap_statistics(tensor: Any) -> dict[str, float]:
    torch = _torch()
    values = tensor.detach().float()
    q = torch.quantile(
        values.flatten(),
        torch.tensor([0.5, 0.9, 0.95, 0.99], device=values.device),
    )
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "median": float(q[0].item()),
        "p90": float(q[1].item()),
        "p95": float(q[2].item()),
        "p99": float(q[3].item()),
        "max": float(values.max().item()),
    }


def _compatible(reference: FeatureArtifact, current: FeatureArtifact) -> None:
    mismatch = reference.metadata.signature.compatibility_error(current.metadata.signature)
    if mismatch:
        raise FeatureSpaceMismatch(
            "Cannot compare:\n"
            f"{reference.metadata.signature.display()}\n\nwith:\n"
            f"{current.metadata.signature.display()}\n"
            f"Incompatible fields: {mismatch}"
        )
    if reference.metadata.device != current.metadata.device:
        raise FeatureSpaceMismatch(
            f"Cannot compare feature devices {reference.metadata.device} and {current.metadata.device}"
        )
    if tuple(reference.tensor.shape) != tuple(current.tensor.shape):
        raise FeatureSpaceMismatch(
            f"Cannot compare tensor shapes {tuple(reference.tensor.shape)} and {tuple(current.tensor.shape)}"
        )


def _metadata(artifact: FeatureArtifact, tensor: Any, operation: str):
    return artifact.metadata.model_copy(update={
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "memory_bytes": tensor.numel() * tensor.element_size(),
        "history": [*artifact.metadata.history, operation],
    })


def normalize(artifact: FeatureArtifact, epsilon: float = 1e-8, mode: str = "l2_channel") -> FeatureArtifact:
    torch = _torch()
    if mode == "none":
        tensor = artifact.tensor.clone()
    elif mode == "l2_channel":
        tensor = artifact.tensor / (
            torch.linalg.vector_norm(artifact.tensor, dim=-1, keepdim=True) + epsilon
        )
    else:
        raise ValueError(f"Unknown feature normalization mode: {mode}")
    return FeatureArtifact(tensor, _metadata(artifact, tensor, f"normalize:{mode}"), tensor_summary(tensor))


def _heatmap(value: Any, artifact: FeatureArtifact, mode: str) -> HeatmapValue:
    metadata = artifact.metadata
    return HeatmapValue(
        tensor=value,
        feature_width=metadata.feature_width,
        feature_height=metadata.feature_height,
        source_width=metadata.source_width,
        source_height=metadata.source_height,
        mode=mode,
        statistics=heatmap_statistics(value),
    )


def difference(reference: FeatureArtifact, current: FeatureArtifact, mode: str) -> tuple[FeatureArtifact, HeatmapValue]:
    _compatible(reference, current)
    torch = _torch()
    raw = current.tensor - reference.tensor
    if mode == "raw":
        delta, heat = raw, raw.abs().mean(dim=-1)
    elif mode == "absolute":
        delta, heat = raw.abs(), raw.abs().mean(dim=-1)
    elif mode == "l2":
        delta, heat = raw, torch.linalg.vector_norm(raw, dim=-1)
    elif mode == "cosine":
        delta = raw
        heat = 1.0 - torch.nn.functional.cosine_similarity(
            current.tensor, reference.tensor, dim=-1, eps=1e-8
        )
    else:
        raise ValueError(f"Unknown feature difference mode: {mode}")
    artifact = FeatureArtifact(delta, _metadata(current, delta, f"difference:{mode}"), tensor_summary(delta))
    return artifact, _heatmap(heat, current, mode)


def activation_xor(reference: FeatureArtifact, current: FeatureArtifact, percentile: float = 90.0) -> HeatmapValue:
    _compatible(reference, current)
    signature = reference.metadata.signature
    if signature.extractor_family != "rfdetr" or signature.extractor_version != "v2.2":
        raise FeatureSpaceMismatch("RF-DETR Activation XOR accepts only rfdetr/v2.2 feature spaces")
    torch = _torch()
    from inference_studio.features.extractors.vendor.rfdetr_feature_xor import reference_channel_thresholds

    # The v2.2 formula: one spatial percentile threshold per REFERENCE channel (frozen; the vendored source
    # function, which also enforces 0 < percentile < 100), applied to both tensors. Computed on-device; the
    # source's fixed_threshold_feature_xor_maps returns the same map on the CPU.
    reference_bchw = reference.tensor.permute(2, 0, 1).unsqueeze(0)
    current_bchw = current.tensor.permute(2, 0, 1).unsqueeze(0)
    (threshold,) = reference_channel_thresholds([reference_bchw], percentile)
    with torch.inference_mode():
        xor = torch.logical_xor(current_bchw.float() > threshold, reference_bchw.float() > threshold).float().mean(dim=1)[0]
    return _heatmap(xor, current, "activation_xor")


@dataclass(frozen=True)
class MahalanobisGeometry:
    mu_clear: Any
    cholesky: Any
    signature: Any
    calibration_corpus: str
    fitting_method: str
    metadata: dict[str, Any]


def load_mahalanobis_geometry(path: str | Path) -> MahalanobisGeometry:
    torch = _torch()
    source = Path(path)
    if not source.is_file():
        raise MahGeometryMismatch(f"Mahalanobis geometry not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise MahGeometryMismatch("Mahalanobis geometry must be a dictionary artifact")
    try:
        from inference_studio.features.schema import FeatureSpaceSignature
        signature = FeatureSpaceSignature.model_validate(payload["feature_signature"])
        metadata = dict(payload.get("metadata", {}))
        return MahalanobisGeometry(
            mu_clear=payload["mu_clear"],
            cholesky=payload["cholesky"],
            signature=signature,
            calibration_corpus=str(metadata["calibration_corpus"]),
            fitting_method=str(metadata["fitting_method"]),
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MahGeometryMismatch(f"Invalid Mahalanobis geometry: {exc}") from exc


def mahalanobis_difference(
    reference: FeatureArtifact,
    current: FeatureArtifact,
    geometry: MahalanobisGeometry,
) -> HeatmapValue:
    _compatible(reference, current)
    signature = reference.metadata.signature
    if not (
        signature.extractor_family == "rfdetr"
        and signature.extractor_version == "architecture-b"
        and (signature.spatial_height, signature.spatial_width, signature.channels) == (42, 42, 256)
    ):
        raise FeatureSpaceMismatch("Mahalanobis Feature Difference accepts only Architecture B 42x42x256 features")
    mismatch = signature.compatibility_error(geometry.signature)
    if mismatch:
        raise MahGeometryMismatch(
            f"Mahalanobis geometry does not match {signature.display()}: {mismatch}"
        )
    torch = _torch()
    # Same precision chain as B1's covariance.mahalanobis_map: float32 delta (Bt - Br), float64 centering and
    # triangular solve against the Cholesky factor (never an explicit inverse), float32 score. The covariance's
    # condition number (~9e5) makes a float32 solve numerically different.
    delta = current.tensor.float() - reference.tensor.float()
    channels = delta.shape[-1]
    mu = torch.as_tensor(geometry.mu_clear, device=delta.device, dtype=torch.float64).reshape(1, -1)
    lower = torch.as_tensor(geometry.cholesky, device=delta.device, dtype=torch.float64)
    if mu.shape[-1] != channels or tuple(lower.shape) != (channels, channels):
        raise MahGeometryMismatch(
            f"Geometry shape mismatch: mu={tuple(mu.shape)}, cholesky={tuple(lower.shape)}, channels={channels}"
        )
    centered = (delta.reshape(-1, channels).to(torch.float64) - mu).T  # (256, N)
    y = torch.linalg.solve_triangular(lower, centered, upper=False)
    score = torch.sum(y.square(), dim=0).reshape(delta.shape[:2]).to(torch.float32)
    return _heatmap(score, current, "mahalanobis")
