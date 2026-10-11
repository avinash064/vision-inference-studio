"""RF-DETR Architecture B feature provider (RFDETRSegLarge, 504x504, patch size 12, 256x42x42 backbone tap).

Model loading, preprocessing and the backbone pass run through the vendored RSD-V1 extractor
(``vendor/rsd_v1_extract_rfdetr_features.py``), the code Architecture B / B1 / B2 used. One backbone forward
yields ``srcs, masks, poss``; the feature is ``srcs[0]`` and the same tensors are kept on the artifact so the frozen
segmentation head (``semantic_outputs``) can later reuse them without a second backbone pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from inference_studio.features.ops import tensor_summary
from inference_studio.features.schema import FeatureMetadata, FeatureSpaceSignature
from inference_studio.runtime.artifacts import FeatureArtifact

from .base import FeatureExtractor, FeatureModelKey, LoadedFeatureModel

PREPROCESSING_SIGNATURE = "rfdetr-arch-b-seglarge-504"
# Exactly what vendor.preprocess_image does (RFDETR.predict()'s own preprocessing).
NORMALIZATION = (
    "bgr->rgb, float/255, torchvision F.resize(504x504), "
    "F.normalize(mean=0.485,0.456,0.406; std=0.229,0.224,0.225)"
)


def architecture_b_signature(checkpoint_sha256: str, stage: str = "backbone_42") -> FeatureSpaceSignature:
    """The feature-space identity of an Architecture B checkpoint (also used to stamp frozen MAH geometry)."""

    return FeatureSpaceSignature(
        extractor_family="rfdetr", extractor_version="architecture-b", extractor_variant="seglarge",
        checkpoint_sha256=checkpoint_sha256, feature_stage=stage,
        spatial_height=42, spatial_width=42, channels=256,
        input_height=504, input_width=504, stride=12.0, patch_size=12,
        preprocessing_signature=PREPROCESSING_SIGNATURE, normalization=NORMALIZATION,
    )


@dataclass
class RFDetrArchBForward:
    """Everything one Architecture B backbone forward produced."""

    feature: Any  # canonical [42, 42, 256] view-copy of srcs[0][0]
    srcs: Any
    masks: Any
    poss: Any
    semantic_output: Any | None = None

    def tensors(self) -> list[Any]:
        return [*self.srcs, *self.masks, *self.poss]


class RFDetrArchBFeatureExtractor(FeatureExtractor):
    name = "rfdetr-arch-b"
    family = "rfdetr"
    version = "architecture-b"
    variant = "seglarge"
    STAGES = {"backbone_42": (42, 42, 256)}
    RESOLUTION = 504
    PATCH_SIZE = 12
    TAP = "RFDETRSegLarge.model.model.backbone(NestedTensor) -> features[0].decompose() = srcs[0] (vendored RSD-V1 extract_backbone_feature)"

    @staticmethod
    def _vendor():
        from .vendor import rsd_v1_extract_rfdetr_features as vendor
        return vendor

    def load(self, key: FeatureModelKey, weight_path: Path, checkpoint_sha256: str) -> LoadedFeatureModel:
        if key.precision == "fp16" and not key.device.startswith("cuda"):
            raise ValueError("Architecture B fp16 extraction requires CUDA")
        try:
            vendor = self._vendor()
        except ImportError as exc:
            raise RuntimeError("Architecture B requires RF-DETR with RFDETRSegLarge") from exc
        frozen = vendor.load_frozen_model(checkpoint_path=str(weight_path), device=key.device)
        if vendor.RESOLUTION != self.RESOLUTION or vendor.RESOLUTION // vendor.FEATURE_SPATIAL != self.PATCH_SIZE:
            raise RuntimeError("Vendored Architecture B constants no longer match 504 px / patch 12")
        return LoadedFeatureModel(frozen, key, weight_path, checkpoint_sha256)

    def stages(self, loaded_model: LoadedFeatureModel) -> list[str]:
        return list(self.STAGES)

    def forward(self, loaded_model: LoadedFeatureModel, image: np.ndarray) -> tuple[RFDetrArchBForward, float, float]:
        """One frozen backbone pass; returns the shared forward plus preprocess/forward milliseconds."""

        import torch

        vendor = self._vendor()
        frozen = loaded_model.handle
        started = perf_counter()
        tensor, _ = vendor.preprocess_image(image, self.RESOLUTION, device=frozen.device)
        preprocess_ms = (perf_counter() - started) * 1000
        half = loaded_model.key.precision == "fp16"
        started = perf_counter()
        # The source runs fp32; fp16 executes the identical backbone call under CUDA autocast.
        with torch.autocast("cuda", dtype=torch.float16, enabled=half):
            srcs, masks, poss = vendor.extract_backbone_feature(frozen, tensor)
        if str(frozen.device).startswith("cuda"):
            torch.cuda.synchronize(frozen.device)
        forward_ms = (perf_counter() - started) * 1000
        feature = srcs[0][0].permute(1, 2, 0).contiguous()  # [1,256,42,42] -> [42,42,256]
        return RFDetrArchBForward(feature, srcs, masks, poss), preprocess_ms, forward_ms

    def semantic_outputs(self, loaded_model: LoadedFeatureModel, forward: RFDetrArchBForward) -> dict[str, Any]:
        """Frozen decoder + segmentation head from the SAME backbone tensors (skip_car / wheel / cable_wire).

        Prepared for semantic suppression; no graph node consumes it yet."""

        vendor = self._vendor()
        if forward.semantic_output is None:
            forward.semantic_output = vendor.run_heads_from_srcs(
                loaded_model.handle, forward.srcs, forward.masks, forward.poss, (self.RESOLUTION, self.RESOLUTION)
            )
        return forward.semantic_output

    def extract(self, loaded_model: LoadedFeatureModel, image: np.ndarray, stage: str) -> FeatureArtifact:
        if stage not in self.STAGES:
            raise ValueError(f"Unsupported Architecture B stage: {stage}")
        source_height, source_width = image.shape[:2]
        forward, preprocess_ms, forward_ms = self.forward(loaded_model, image)
        feature = forward.feature
        height, width, channels = self.STAGES[stage]
        if tuple(feature.shape) != (height, width, channels):
            raise RuntimeError(f"Unexpected Architecture B feature shape: {tuple(feature.shape)}")
        signature = architecture_b_signature(loaded_model.checkpoint_sha256, stage)
        metadata = FeatureMetadata(
            shape=list(feature.shape), dtype=str(feature.dtype).removeprefix("torch."), device=str(feature.device),
            memory_bytes=feature.numel() * feature.element_size(),
            source_width=source_width, source_height=source_height,
            feature_width=width, feature_height=height, channels=channels,
            extractor_family="rfdetr", extractor_version="architecture-b", extractor_variant="seglarge",
            feature_stage=stage, stride=float(self.PATCH_SIZE), patch_size=self.PATCH_SIZE, signature=signature,
        )
        shared_bytes = sum(t.numel() * t.element_size() for t in forward.tensors() if hasattr(t, "numel"))
        return FeatureArtifact(
            feature, metadata, tensor_summary(feature),
            metrics={
                "feature_preprocess_ms": preprocess_ms,
                "feature_forward_ms": forward_ms,
                "feature_extract_ms": preprocess_ms + forward_ms,
                "tap": self.TAP,
                "shared_forward_bytes": shared_bytes,
            },
            shared_forward=forward,
        )
