from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference_studio.adapters import ModelConfig
from inference_studio.assets import AssetRecord
from inference_studio.results import InferenceResult


@dataclass
class ImageValue:
    image: np.ndarray
    asset: AssetRecord | None = None


@dataclass
class ModelOutput:
    image: np.ndarray
    result: InferenceResult
    config: ModelConfig


@dataclass
class VideoValue:
    asset: AssetRecord
    frame_stride: int = 1
    warmup_frames: int = 2
    benchmark_frames: int = 0


@dataclass
class VideoModelStream:
    asset: AssetRecord
    config: ModelConfig
    frame_stride: int = 1
    warmup_frames: int = 2
    benchmark_frames: int = 0
