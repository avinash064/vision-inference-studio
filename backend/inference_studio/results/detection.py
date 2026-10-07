from __future__ import annotations

from pydantic import BaseModel, Field

from .base import InferenceMetrics, InferenceResult


class Detection(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    class_id: int
    label: str
    confidence: float = Field(ge=0, le=1)


class DetectionResult(InferenceResult):
    result_type: str = "detections"
    detections: list[Detection] = Field(default_factory=list)
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)

