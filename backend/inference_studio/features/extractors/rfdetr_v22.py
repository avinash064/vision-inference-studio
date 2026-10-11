"""RF-DETR v2.2 reference-feature provider (RFDETRLargeDeprecated backbone, 560x560).

Model loading, preprocessing, the backbone forward hook and the shape-based level discovery all run through the
vendored v2.2 runtime (``vendor/rfdetr_xor_runtime.py``); this module only selects a level, converts it to the
canonical ``[H, W, C]`` layout and attaches the feature-space signature.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from inference_studio.features.ops import tensor_summary
from inference_studio.features.schema import FeatureMetadata, FeatureSpaceSignature
from inference_studio.runtime.artifacts import FeatureArtifact

from .base import FeatureExtractor, FeatureModelKey, LoadedFeatureModel

PREPROCESSING_SIGNATURE = "rfdetr-v22-560-rgb01-imagenet"
INPUT_SIZE = (560, 560)  # (width, height), as the v2.2 runtime's model_size
STRIDES = {"fine_80": 7.0, "coarse_20": 28.0}


def _timed_boundary_features(weight_path: Path, device: str):
    """The vendored extractor, with its own ``_preprocess`` timed (logic unchanged)."""

    import torch

    from .vendor.rfdetr_xor_runtime import RFDETRBoundaryFeatures

    class TimedBoundaryFeatures(RFDETRBoundaryFeatures):
        last_preprocess_ms = 0.0

        def _preprocess(self, bgr: np.ndarray) -> Any:
            started = perf_counter()
            nested = super()._preprocess(bgr)
            self.last_preprocess_ms = (perf_counter() - started) * 1000
            return nested

    return TimedBoundaryFeatures(weight_path, torch.device(device), INPUT_SIZE)


class RFDetrV22FeatureExtractor(FeatureExtractor):
    name = "rfdetr-v2.2"
    family = "rfdetr"
    version = "v2.2"
    variant = "RFDETRLargeDeprecated"
    STAGES = {
        "fine_80": (80, 80, 384),
        "coarse_20": (20, 20, 384),
    }
    TAP = "RFDETRLargeDeprecated.model.model.backbone forward hook (vendored v2.2 runtime), level selected by shape"

    def load(self, key: FeatureModelKey, weight_path: Path, checkpoint_sha256: str) -> LoadedFeatureModel:
        try:
            import torch  # noqa: F401
            import rfdetr  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("RF-DETR v2.2 feature runtime requires rfdetr and PyTorch") from exc
        if key.precision == "fp16" and not key.device.startswith("cuda"):
            raise ValueError("RF-DETR fp16 extraction requires CUDA")
        return LoadedFeatureModel(_timed_boundary_features(weight_path, key.device), key, weight_path, checkpoint_sha256)

    def stages(self, loaded_model: LoadedFeatureModel) -> list[str]:
        return list(self.STAGES)

    def extract(self, loaded_model: LoadedFeatureModel, image: np.ndarray, stage: str) -> FeatureArtifact:
        if stage not in self.STAGES:
            raise ValueError(f"Unsupported RF-DETR v2.2 stage: {stage}")
        import torch

        features = loaded_model.handle
        device = loaded_model.key.device
        source_height, source_width = image.shape[:2]
        half = loaded_model.key.precision == "fp16"
        started = perf_counter()
        # The v2.2 runtime is fp32; fp16 runs the identical code under CUDA autocast.
        with torch.autocast("cuda", dtype=torch.float16, enabled=half):
            levels = features(image)  # finest -> coarsest, detached, still on `device`
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        call_ms = (perf_counter() - started) * 1000
        preprocess_ms = features.last_preprocess_ms

        height, width, channels = self.STAGES[stage]
        raw = next((level for level in levels if tuple(level.shape[1:]) == (channels, height, width)), None)
        if raw is None:
            discovered = [tuple(int(v) for v in level.shape[1:]) for level in levels]
            raise RuntimeError(f"RF-DETR v2.2 did not expose {stage}={self.STAGES[stage]} as [C,H,W]; discovered {discovered}")
        feature = raw[0].permute(1, 2, 0).contiguous()  # [B,C,H,W] -> batch 0 -> [H,W,C]
        del levels, raw
        if tuple(feature.shape) != self.STAGES[stage]:
            raise RuntimeError(f"Unexpected RF-DETR v2.2 {stage} shape: {tuple(feature.shape)}")

        stride = STRIDES[stage]
        signature = FeatureSpaceSignature(
            extractor_family="rfdetr", extractor_version="v2.2", extractor_variant=self.variant,
            checkpoint_sha256=loaded_model.checkpoint_sha256, feature_stage=stage,
            spatial_height=height, spatial_width=width, channels=channels,
            input_height=INPUT_SIZE[1], input_width=INPUT_SIZE[0], stride=stride, patch_size=None,
            preprocessing_signature=PREPROCESSING_SIGNATURE, normalization="imagenet",
        )
        metadata = FeatureMetadata(
            shape=list(feature.shape), dtype=str(feature.dtype).removeprefix("torch."), device=str(feature.device),
            memory_bytes=feature.numel() * feature.element_size(),
            source_width=source_width, source_height=source_height,
            feature_width=width, feature_height=height, channels=channels,
            extractor_family="rfdetr", extractor_version="v2.2", extractor_variant=self.variant,
            feature_stage=stage, stride=stride, signature=signature,
        )
        return FeatureArtifact(
            feature, metadata, tensor_summary(feature),
            metrics={
                "feature_preprocess_ms": preprocess_ms,
                "feature_forward_ms": max(call_ms - preprocess_ms, 0.0),
                "feature_extract_ms": call_ms,
                "tap": self.TAP,
                "levels": features.level_metadata,
            },
        )
