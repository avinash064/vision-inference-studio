from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ShapeList(list[int]):
    """JSON list with sequence-friendly equality for v0.5 tuple callers."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return tuple(self) == other
        return super().__eq__(other)


class FeatureSpaceSignature(BaseModel):
    """Complete identity of a feature space; every field is compatible-or-not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extractor_family: str
    extractor_version: str
    extractor_variant: str
    checkpoint_sha256: str
    feature_stage: str
    tensor_layout: str = "HWC"
    spatial_height: int
    spatial_width: int
    channels: int
    input_height: int
    input_width: int
    stride: float | None = None
    patch_size: int | None = None
    preprocessing_signature: str
    normalization: str

    def __init__(self, *args: Any, **data: Any) -> None:
        if args:
            if data:
                raise TypeError("Use positional legacy fields or keyword fields, not both")
            if len(args) != 14:
                raise TypeError(f"Legacy FeatureSpaceSignature expects 14 values, got {len(args)}")
            (
                family, version, variant, checkpoint, stage, processor,
                input_height, input_width, patch_size, spatial_height,
                spatial_width, channels, normalization, _legacy_dtype,
            ) = args
            data = {
                "extractor_family": family, "extractor_version": version,
                "extractor_variant": variant, "checkpoint_sha256": checkpoint,
                "feature_stage": stage, "spatial_height": spatial_height,
                "spatial_width": spatial_width, "channels": channels,
                "input_height": input_height, "input_width": input_width,
                "stride": float(patch_size), "patch_size": patch_size,
                "preprocessing_signature": processor, "normalization": normalization,
            }
        super().__init__(**data)

    def compatibility_error(self, other: "FeatureSpaceSignature") -> str | None:
        mismatches = [name for name in type(self).model_fields if getattr(self, name) != getattr(other, name)]
        return ", ".join(mismatches) if mismatches else None

    def display(self) -> str:
        return (
            f"{self.extractor_family}/{self.extractor_version}/"
            f"{self.extractor_variant}/{self.feature_stage} "
            f"[{self.spatial_height},{self.spatial_width},{self.channels}]"
        )

    # Compatibility properties for pre-v0.6 internal callers.
    @property
    def family(self) -> str: return self.extractor_family
    @property
    def version(self) -> str: return self.extractor_version
    @property
    def size(self) -> str: return self.extractor_variant
    @property
    def layer(self) -> str: return self.feature_stage
    @property
    def processor(self) -> str: return self.preprocessing_signature
    @property
    def grid_height(self) -> int: return self.spatial_height
    @property
    def grid_width(self) -> int: return self.spatial_width


class FeatureMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shape: list[int]
    dtype: str
    device: str
    memory_bytes: int
    source_width: int
    source_height: int
    feature_width: int
    feature_height: int
    channels: int
    extractor_family: str
    extractor_version: str
    extractor_variant: str
    feature_stage: str
    stride: float | None = None
    patch_size: int | None = None
    signature: FeatureSpaceSignature
    history: list[str] = Field(default_factory=list)

    def __init__(self, *args: Any, **data: Any) -> None:
        if args:
            if data:
                raise TypeError("Use positional legacy fields or keyword fields, not both")
            if len(args) != 15:
                raise TypeError(f"Legacy FeatureMetadata expects 15 values, got {len(args)}")
            (
                shape, dtype, device, memory_bytes, source_height, source_width,
                feature_height, feature_width, channels, stage, _token_type,
                family, version, variant, signature,
            ) = args
            data = {
                "shape": list(shape), "dtype": dtype, "device": device,
                "memory_bytes": memory_bytes, "source_width": source_width,
                "source_height": source_height, "feature_width": feature_width,
                "feature_height": feature_height, "channels": channels,
                "extractor_family": family, "extractor_version": version,
                "extractor_variant": variant, "feature_stage": stage,
                "stride": signature.stride, "patch_size": signature.patch_size,
                "signature": signature,
            }
        super().__init__(**data)
        object.__setattr__(self, "shape", _ShapeList(self.shape))

    def public(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def grid_height(self) -> int: return self.feature_height
    @property
    def grid_width(self) -> int: return self.feature_width
    @property
    def layer(self) -> str: return self.feature_stage
    @property
    def token_type(self) -> str: return "patch"
    @property
    def model_family(self) -> str: return self.extractor_family
    @property
    def model_version(self) -> str: return self.extractor_version
    @property
    def model_size(self) -> str: return self.extractor_variant
    @property
    def space(self) -> FeatureSpaceSignature: return self.signature
    @property
    def operation_history(self) -> tuple[str, ...]: return tuple(self.history)


class HeatmapMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_width: int
    feature_height: int
    source_width: int
    source_height: int
    device: str
    mode: Literal["raw", "absolute", "l2", "cosine", "activation_xor", "mahalanobis"]
    statistics: dict[str, float]
