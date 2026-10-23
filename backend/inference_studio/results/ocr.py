from __future__ import annotations

from pydantic import BaseModel, Field

from .base import InferenceMetrics, InferenceResult


class OcrSpan(BaseModel):
    text: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)


class OcrResult(InferenceResult):
    result_type: str = "ocr"
    text: str = ""
    spans: list[OcrSpan] = Field(default_factory=list)
    language: str | None = None
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)
