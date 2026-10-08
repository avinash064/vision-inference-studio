from __future__ import annotations

from pydantic import BaseModel, Field

from .base import InferenceMetrics, InferenceResult


class SegmentationMask(BaseModel):
    """A JSON-safe instance mask represented by one or more polygons."""

    class_id: int
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    area: int = Field(ge=0)
    polygons: list[list[float]] = Field(default_factory=list)


class SegmentationResult(InferenceResult):
    result_type: str = "segmentation"
    masks: list[SegmentationMask] = Field(default_factory=list)
    prompt_mode: str
    prompt: str
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)
