from __future__ import annotations

from pydantic import Field

from .base import InferenceMetrics, InferenceResult


class TensorResult(InferenceResult):
    result_type: str = "tensor"
    shape: list[int]
    dtype: str
    values: list[float] = Field(default_factory=list)
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)

