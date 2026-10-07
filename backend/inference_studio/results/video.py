from __future__ import annotations

from pydantic import BaseModel, SerializeAsAny

from .base import InferenceResult


class VideoFrameResult(BaseModel):
    frame_index: int
    timestamp_seconds: float
    result: SerializeAsAny[InferenceResult]
