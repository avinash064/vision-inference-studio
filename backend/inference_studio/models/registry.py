from __future__ import annotations

from pathlib import Path

import yaml

from .manifest import ModelFamily, ModelManifest


class ModelRegistryError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest

    @classmethod
    def load(cls, path: Path) -> "ModelRegistry":
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return cls(ModelManifest.model_validate(data))
        except Exception as exc:
            raise ModelRegistryError(f"Invalid model registry: {path}") from exc

    def get(self, family: str) -> ModelFamily:
        try:
            return self.manifest.families[family]
        except KeyError as exc:
            raise ModelRegistryError(f"Unknown model family: {family}") from exc

    def validate_selection(self, family: str, version: str, size: str) -> ModelFamily:
        definition = self.get(family)
        if version not in definition.versions:
            raise ModelRegistryError(f"Unknown {family} version: {version}")
        if size not in definition.versions[version].sizes:
            raise ModelRegistryError(f"Unsupported {family} size: {size}")
        return definition

    def public_data(self) -> dict:
        return self.manifest.model_dump()

