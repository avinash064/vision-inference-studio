from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InferenceMetrics(BaseModel):
    preprocess_ms: float = 0
    inference_ms: float = 0
    postprocess_ms: float = 0
    total_ms: float = 0
    gpu_memory_mb: float = 0
    gpu_reserved_mb: float = 0
    gpu_peak_memory_mb: float = 0
    device_name: str | None = None


class InferenceResult(BaseModel):
    result_type: str
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)


ResultType = Literal[
    "detections",
    "feature_map",
    "embedding",
    "segmentation",
    "tensor",
    "heatmap",
    "ocr",
]
