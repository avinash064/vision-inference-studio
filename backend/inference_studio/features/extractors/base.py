from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from inference_studio.runtime.artifacts import FeatureArtifact


@dataclass(frozen=True)
class FeatureModelKey:
    family: str
    version: str
    variant: str
    weight_id: str
    device: str
    precision: Literal["fp32", "fp16"] = "fp32"


@dataclass
class LoadedFeatureModel:
    handle: Any
    key: FeatureModelKey
    weight_path: Path
    checkpoint_sha256: str


class FeatureExtractor(ABC):
    name: str
    family: str
    version: str
    variant: str

    def model_key(self, weight_id: str, device: str, precision: str) -> FeatureModelKey:
        return FeatureModelKey(
            self.family, self.version, self.variant, weight_id, device, precision  # type: ignore[arg-type]
        )

    @abstractmethod
    def load(self, key: FeatureModelKey, weight_path: Path, checkpoint_sha256: str) -> LoadedFeatureModel:
        ...

    @abstractmethod
    def stages(self, loaded_model: LoadedFeatureModel) -> list[str]:
        ...

    @abstractmethod
    def extract(self, loaded_model: LoadedFeatureModel, image: Any, stage: str) -> FeatureArtifact:
        ...

    def unload(self, loaded_model: LoadedFeatureModel) -> None:
        del loaded_model.handle
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
