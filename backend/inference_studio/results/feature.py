from __future__ import annotations

from pydantic import Field

from .base import InferenceMetrics, InferenceResult


class FeatureResult(InferenceResult):
    result_type: str = "feature_map"
    shape: list[int]
    values: list[float] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)
    feature_id: str | None = None
    dtype: str | None = None
    device: str | None = None
    memory_bytes: int = 0
    metadata: dict = Field(default_factory=dict)
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)
