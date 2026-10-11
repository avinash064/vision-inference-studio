from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from inference_studio.assets import AssetRecord
from inference_studio.results import InferenceResult
from inference_studio.features.schema import FeatureMetadata

if TYPE_CHECKING:
    from inference_studio.adapters import ModelConfig


class RuntimeValue:
    """Marker for values whose graph semantics must not be inferred from shape."""


@dataclass
class ImageValue(RuntimeValue):
    image: np.ndarray
    asset: AssetRecord | None = None


@dataclass
class TensorValue(RuntimeValue):
    value: Any
    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"

    @property
    def tensor(self) -> Any:
        return self.value


@dataclass(frozen=True)
class FeatureValue(RuntimeValue):
    feature_id: str
    metadata: FeatureMetadata
    summary: dict[str, float] = field(default_factory=dict)


@dataclass
class HeatmapValue(RuntimeValue):
    tensor: Any
    feature_width: int
    feature_height: int
    source_width: int
    source_height: int
    mode: str
    statistics: dict[str, float]

    @property
    def values(self) -> Any:
        return self.tensor

    @property
    def grid_width(self) -> int:
        return self.feature_width

    @property
    def grid_height(self) -> int:
        return self.feature_height


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


@dataclass(frozen=True)
class ScalarValue(RuntimeValue):
    value: str | int | float | bool


@dataclass(frozen=True)
class FileValue(RuntimeValue):
    path: str


@dataclass(frozen=True)
class DirectoryValue(RuntimeValue):
    path: str


@dataclass
class ObjectValue(RuntimeValue):
    value: Any
