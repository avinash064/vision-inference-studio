from __future__ import annotations

from pathlib import Path
from typing import Any

from inference_studio.adapters.base import InferenceRequest, ModelKey
from inference_studio.adapters.dinov3 import DinoV3Adapter
from inference_studio.runtime.artifacts import FeatureArtifact

from .base import FeatureExtractor, FeatureModelKey, LoadedFeatureModel


class DinoV3FeatureExtractor(FeatureExtractor):
    name = "dinov3"
    family = "dinov3"
    version = "official"
    variant = "vit-b"

    def __init__(self) -> None:
        self.adapter = DinoV3Adapter()

    def load(self, key: FeatureModelKey, weight_path: Path, checkpoint_sha256: str) -> LoadedFeatureModel:
        adapter_key = ModelKey(key.family, key.version, key.variant, key.weight_id, key.device, key.precision)
        loaded = self.adapter.load(adapter_key, weight_path)
        loaded.checkpoint_sha256 = checkpoint_sha256
        return LoadedFeatureModel(loaded, key, weight_path, checkpoint_sha256)

    def stages(self, loaded_model: LoadedFeatureModel) -> list[str]:
        return ["last_hidden_state"]

    def extract(self, loaded_model: LoadedFeatureModel, image: Any, stage: str) -> FeatureArtifact:
        if stage != "last_hidden_state":
            raise ValueError(f"Unsupported DINOv3 feature stage: {stage}")
        return self.adapter.predict(loaded_model.handle, InferenceRequest(), image)

    def unload(self, loaded_model: LoadedFeatureModel) -> None:
        self.adapter.unload(loaded_model.handle)
