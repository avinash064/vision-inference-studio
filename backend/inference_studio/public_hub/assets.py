from __future__ import annotations

import uuid
from pathlib import Path

from inference_studio.durable import JsonDocumentStore

from .integrity import tree_identity
from .schemas import DatasetAsset, DatasetInspection, ExternalResource, ModelArtifact, ModelInspection, Provenance


class ImportedAssetRegistry:
    def __init__(self, root: Path) -> None:
        self.datasets = JsonDocumentStore(root / "datasets", "ds_")
        self.models = JsonDocumentStore(root / "models", "mdl_")

    @staticmethod
    def _source(resource: ExternalResource) -> Provenance:
        return Provenance(provider=resource.ref.provider, resource_id=resource.ref.resource_id,
            requested_version=resource.ref.version, requested_revision=resource.ref.revision,
            resolved_revision=resource.resolved_revision, source_url=resource.source_url,
            owner=resource.owner, license=resource.license)

    def register_dataset(self, root: Path, resource: ExternalResource, inspection: DatasetInspection) -> DatasetAsset:
        checksum, size = tree_identity(root)
        for raw in self.datasets.list():
            if raw.get("checksum") == checksum:
                raw["local_root"] = str(root)
                return DatasetAsset.model_validate(raw)
        asset = DatasetAsset(dataset_id=f"ds_{uuid.uuid4().hex}", display_name=resource.title, provenance=self._source(resource), local_root=str(root), format=inspection.format, task=inspection.task, status=inspection.state, image_count=inspection.valid_image_count, annotation_count=inspection.annotation_count, classes=inspection.classes, splits=inspection.splits, checksum=checksum, size_bytes=size, inspection=inspection)
        self.datasets.put(asset.dataset_id, asset.model_dump(mode="json"))
        return asset

    def register_model(self, root: Path, resource: ExternalResource, inspection: ModelInspection, weight_id: str | None = None) -> ModelArtifact:
        checksum, size = tree_identity(root)
        for raw in self.models.list():
            if raw.get("checksum") == checksum:
                raw["local_root"] = str(root)
                return ModelArtifact.model_validate(raw)
        asset = ModelArtifact(model_artifact_id=f"mdl_{uuid.uuid4().hex}", display_name=resource.title, provenance=self._source(resource), local_root=str(root), status=inspection.state, framework=inspection.framework, architecture=inspection.architecture, candidate_adapter=inspection.candidate_adapter, weight_id=weight_id, artifact_file=inspection.artifact_file, checksum=checksum, size_bytes=size, inspection=inspection)
        self.models.put(asset.model_artifact_id, asset.model_dump(mode="json"))
        return asset

    def list_datasets(self) -> list[dict]:
        return self.datasets.list()

    def list_models(self) -> list[dict]:
        return self.models.list()

    def get_dataset(self, identity: str) -> DatasetAsset:
        return DatasetAsset.model_validate(self.datasets.get(identity))

    def get_model(self, identity: str) -> ModelArtifact:
        return ModelArtifact.model_validate(self.models.get(identity))
