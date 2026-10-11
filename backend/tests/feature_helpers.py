"""Builders for synthetic feature artifacts that carry the real RF-DETR provider signatures."""

from __future__ import annotations

import torch

from inference_studio.features.extractors.rfdetr_arch_b import architecture_b_signature
from inference_studio.features.extractors.rfdetr_v22 import PREPROCESSING_SIGNATURE as V22_PREPROCESSING, STRIDES
from inference_studio.features.ops import tensor_summary
from inference_studio.features.schema import FeatureMetadata, FeatureSpaceSignature
from inference_studio.runtime.artifacts import FeatureArtifact

CKPT_A = "a" * 64
CKPT_B = "b" * 64
V22_SHAPES = {"fine_80": (80, 80, 384), "coarse_20": (20, 20, 384)}


def v22_signature(stage: str = "fine_80", checkpoint: str = CKPT_A) -> FeatureSpaceSignature:
    height, width, channels = V22_SHAPES[stage]
    return FeatureSpaceSignature(
        extractor_family="rfdetr", extractor_version="v2.2", extractor_variant="RFDETRLargeDeprecated",
        checkpoint_sha256=checkpoint, feature_stage=stage, spatial_height=height, spatial_width=width,
        channels=channels, input_height=560, input_width=560, stride=STRIDES[stage], patch_size=None,
        preprocessing_signature=V22_PREPROCESSING, normalization="imagenet",
    )


def artifact(tensor: torch.Tensor, signature: FeatureSpaceSignature, source=(1080, 1920), shared_forward=None) -> FeatureArtifact:
    height, width, channels = tensor.shape
    metadata = FeatureMetadata(
        shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix("torch."), device=str(tensor.device),
        memory_bytes=tensor.numel() * tensor.element_size(), source_width=source[1], source_height=source[0],
        feature_width=width, feature_height=height, channels=channels,
        extractor_family=signature.extractor_family, extractor_version=signature.extractor_version,
        extractor_variant=signature.extractor_variant, feature_stage=signature.feature_stage,
        stride=signature.stride, patch_size=signature.patch_size, signature=signature,
    )
    return FeatureArtifact(tensor, metadata, tensor_summary(tensor), shared_forward=shared_forward)


def v22(stage: str = "fine_80", checkpoint: str = CKPT_A, seed: int = 0, device: str = "cpu") -> FeatureArtifact:
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.randn(V22_SHAPES[stage], generator=generator).to(device)
    return artifact(tensor, v22_signature(stage, checkpoint))


def arch_b(checkpoint: str = CKPT_A, seed: int = 0, device: str = "cpu") -> FeatureArtifact:
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.randn((42, 42, 256), generator=generator).to(device)
    return artifact(tensor, architecture_b_signature(checkpoint))
