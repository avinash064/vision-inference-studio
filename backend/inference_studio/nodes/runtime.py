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
class VideoModelStream:
    asset: AssetRecord
    config: ModelConfig
