from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference_studio.api import routes as api_routes
from inference_studio.config import Settings
from inference_studio.main import create_app
from inference_studio.models import ModelRegistry, WeightStore
from inference_studio.public_hub.assets import ImportedAssetRegistry
from inference_studio.public_hub.base import DownloadedResource, ExternalProvider
from inference_studio.public_hub.benchmarks import BenchmarkRegistry
from inference_studio.public_hub.downloads import DownloadManager
from inference_studio.public_hub.credentials import ProviderCredentialStore
from inference_studio.public_hub.experiments import ExperimentRegistry
from inference_studio.public_hub.inspectors import DatasetInspector, ModelCompatibilityInspector
from inference_studio.public_hub.integrity import IntegrityError, safe_extract_archive
from inference_studio.public_hub.registry import ExternalProviderRegistry
from inference_studio.public_hub.recipes import RecipeRegistry
from inference_studio.public_hub.schemas import (
    AuthState, BenchmarkCreate, DownloadState, ExternalResource, ExternalResourceRef,
    ExperimentState, InferenceExperiment, PromoteRequest, ProviderCapabilities, ProviderCredentialUpdate, RecipeVariant,
    ResourceDownloadRequest, ResourceKind, SearchPage,
)


class FakeProvider(ExternalProvider):
    name, display_name = "fake", "Fake"
    capabilities = ProviderCapabilities(search_datasets=True, download_datasets=True)

    def __init__(self, fail: bool = False): self.fail = fail
    def auth_state(self): return AuthState.CONFIGURED
    def test_auth(self): return AuthState.CONFIGURED
    def search(self, kind, query, page, page_size):
        item = ExternalResource(ref=ExternalResourceRef(provider="fake", kind=kind, resource_id="owner/resource", revision="abc123"), title="Resource", license="MIT")
        return SearchPage(items=[item], page=page, page_size=page_size, has_more=False)
    def details(self, ref): return ExternalResource(ref=ref, title="Resource", resolved_revision="abc123", license="MIT")
    def download(self, ref, destination, requested_format, progress, cancelled):
        if self.fail: raise RuntimeError("network interrupted")
        destination.mkdir(parents=True)
        archive = destination / "dataset.zip"
        image = np.zeros((8, 8, 3), np.uint8); ok, encoded = cv2.imencode(".jpg", image); assert ok
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("train/images/a.jpg", encoded.tobytes())
            package.writestr("train/labels/a.txt", "0 0.5 0.5 0.5 0.5\n")
            package.writestr("data.yaml", "names: [object]\n")
        progress(archive.stat().st_size, archive.stat().st_size)
        return DownloadedResource(archive, "abc123")


def _settings(tmp_path: Path) -> Settings:
    root = Path(__file__).parents[2]
    return Settings(project_root=root, workspace_dir=tmp_path / "workspace", weights_dir=tmp_path / "weights", workflows_dir=tmp_path / "workflows", benchmarks_dir=tmp_path / "runtime-benchmarks", models_file=root / "config/models.yaml")


def _image(path: Path, value: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((12, 16, 3), value, np.uint8))


def test_provider_registry_discovery_and_capability_gating():
    registry = ExternalProviderRegistry([FakeProvider()])
    assert registry.list()[0]["capabilities"]["search_datasets"] is True
    assert registry.list()[0]["capabilities"]["search_models"] is False
    assert registry.search(ResourceKind.MODEL, "x", None, 1, 20)["items"] == []


def test_provider_status_never_exposes_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "super-secret-token")
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        body = client.get("/api/v1/public-hub/providers").text
    assert "super-secret-token" not in body
    assert "token" not in body.lower()


def test_public_hub_dataset_preview_route_is_bounded_to_huggingface(monkeypatch, tmp_path):
    image = np.full((12, 16, 3), 80, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    monkeypatch.setattr(api_routes, "_huggingface_dataset_preview", lambda resource, revision: encoded.tobytes())
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        preview = client.get("/api/v1/public-hub/dataset-preview", params={"provider": "huggingface", "resource_id": "owner/data", "revision": "abc"})
        unsupported = client.get("/api/v1/public-hub/dataset-preview", params={"provider": "kaggle", "resource_id": "owner/data"})
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    assert preview.headers["x-content-type-options"] == "nosniff"
    assert preview.content.startswith(b"\xff\xd8")
    assert unsupported.status_code == 404


def test_safe_zip_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as package: package.writestr("../../escape.txt", "bad")
    with pytest.raises(IntegrityError, match="Unsafe"):
        safe_extract_archive(archive, tmp_path / "out", 1000, 1000)
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extraction_enforces_member_and_total_limits(tmp_path):
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as package: package.writestr("large.bin", b"x" * 101)
    with pytest.raises(IntegrityError, match="exceeds"):
        safe_extract_archive(archive, tmp_path / "out", 100, 100)


def test_dataset_inspector_detects_coco(tmp_path):
    _image(tmp_path / "images" / "one.jpg")
    (tmp_path / "annotations.json").write_text(json.dumps({"images": [{"id": 1, "file_name": "images/one.jpg"}], "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5]}], "categories": [{"id": 2, "name": "helmet"}]}))
    result = DatasetInspector().inspect(tmp_path)
    assert result.state.value == "READY" and result.format == "COCO" and result.classes == ["helmet"]


def test_dataset_inspector_reports_unsupported_honestly(tmp_path):
    _image(tmp_path / "image.jpg")
    result = DatasetInspector().inspect(tmp_path)
    assert result.state.value == "UNSUPPORTED" and result.task == "unknown"


def test_dataset_inspector_detects_yolo_and_broken_image(tmp_path):
    _image(tmp_path / "images/train/a.jpg")
    (tmp_path / "images/train/b.jpg").write_text("not an image")
    (tmp_path / "labels/train").mkdir(parents=True)
    (tmp_path / "labels/train/a.txt").write_text("0 .5 .5 .2 .2\n")
    (tmp_path / "data.yaml").write_text("names: [person]\n")
    result = DatasetInspector().inspect(tmp_path)
    assert result.format == "YOLO" and result.annotation_count == 1 and result.broken_files


def test_model_inspector_accepts_onnx_and_does_not_execute_python(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"safe placeholder")
    (tmp_path / "evil.py").write_text("raise RuntimeError('executed')")
    result = ModelCompatibilityInspector().inspect(tmp_path)
    assert result.state.value == "READY" and result.candidate_adapter == "onnx"
    assert result.ignored_python_files == ["evil.py"]


def test_unknown_torch_model_requires_adapter(tmp_path):
    (tmp_path / "mystery.pt").write_bytes(b"never unpickle this")
    result = ModelCompatibilityInspector().inspect(tmp_path)
    assert result.state.value == "ADAPTER_REQUIRED" and result.candidate_adapter is None


@pytest.mark.asyncio
async def test_background_download_completes_and_registers(tmp_path):
    settings = _settings(tmp_path); settings.prepare()
    assets = ImportedAssetRegistry(settings.public_hub_root / "metadata/assets")
    manager = DownloadManager(settings, ExternalProviderRegistry([FakeProvider()]), assets)
    job = manager.create(ResourceDownloadRequest(resource=ExternalResourceRef(provider="fake", kind=ResourceKind.DATASET, resource_id="owner/resource")))
    assert job.state == DownloadState.QUEUED
    await manager._tasks[job.download_id]
    completed = manager.get(job.download_id)
    assert completed.state == DownloadState.COMPLETED and completed.asset_id
    assert assets.get_dataset(completed.asset_id).provenance.resolved_revision == "abc123"


@pytest.mark.asyncio
async def test_failed_download_can_retry(tmp_path):
    settings = _settings(tmp_path); settings.prepare(); provider = FakeProvider(fail=True)
    manager = DownloadManager(settings, ExternalProviderRegistry([provider]), ImportedAssetRegistry(settings.public_hub_root / "metadata/assets"))
    job = manager.create(ResourceDownloadRequest(resource=ExternalResourceRef(provider="fake", kind=ResourceKind.DATASET, resource_id="owner/resource")))
    await manager._tasks[job.download_id]
    assert manager.get(job.download_id).state == DownloadState.ERROR
    provider.fail = False; retried = manager.retry(job.download_id); assert retried.retry_count == 1
    await manager._tasks[job.download_id]
    assert manager.get(job.download_id).state == DownloadState.COMPLETED


@pytest.mark.asyncio
async def test_duplicate_active_download_is_reused(tmp_path):
    settings = _settings(tmp_path); settings.prepare()
    manager = DownloadManager(settings, ExternalProviderRegistry([FakeProvider()]), ImportedAssetRegistry(settings.public_hub_root / "metadata/assets"))
    request = ResourceDownloadRequest(resource=ExternalResourceRef(provider="fake", kind=ResourceKind.DATASET, resource_id="same"))
    first, second = manager.create(request), manager.create(request)
    assert first.download_id == second.download_id
    await manager._tasks[first.download_id]


def _registered_dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    for index, cls in enumerate(("a", "a", "b", "b", "c")):
        _image(root / "train" / cls / f"{index}.jpg", index)
    inspection = DatasetInspector().inspect(root)
    registry = ImportedAssetRegistry(tmp_path / "registry")
    resource = ExternalResource(ref=ExternalResourceRef(provider="fake", kind=ResourceKind.DATASET, resource_id="data", revision="1"), title="Data", resolved_revision="1")
    return registry, registry.register_dataset(root, resource, inspection)


def test_same_seed_produces_same_benchmark_selection(tmp_path):
    assets, dataset = _registered_dataset(tmp_path)
    registry = BenchmarkRegistry(tmp_path / "benchmarks", assets)
    request = BenchmarkCreate(name="one", dataset_id=dataset.dataset_id, requested_size=3, seed=9)
    first = registry.create(request); request.name = "two"; second = registry.create(request)
    assert [x["item_id"] for x in first.selected_items] == [x["item_id"] for x in second.selected_items]
    assert first.benchmark_hash == second.benchmark_hash


def test_manual_benchmark_rejects_unknown_items(tmp_path):
    assets, dataset = _registered_dataset(tmp_path)
    registry = BenchmarkRegistry(tmp_path / "benchmarks", assets)
    with pytest.raises(ValueError, match="unknown"):
        registry.create(BenchmarkCreate(name="bad", dataset_id=dataset.dataset_id, sampling_strategy="manual", selected_item_ids=["missing"]))


def test_benchmark_document_is_create_only(tmp_path):
    assets, dataset = _registered_dataset(tmp_path)
    registry = BenchmarkRegistry(tmp_path / "benchmarks", assets)
    benchmark = registry.create(BenchmarkCreate(name="fixed", dataset_id=dataset.dataset_id, requested_size=2))
    with pytest.raises(FileExistsError): registry.store.put(benchmark.benchmark_id, {"changed": True}, create_only=True)


def test_experiment_unit_identity_reuses_exact_configuration():
    item = {"sha256": "image"}; variant = RecipeVariant(variant_id="a", name="A", workflow={"version": "0.2", "name": "a", "nodes": [{"id": "source", "type": "image_source", "config": {}}, {"id": "model", "type": "model", "config": {"weight_id": "w"}}], "edges": []})
    first, hashes = ExperimentRegistry._unit_identity("benchmark", item, "model", variant)
    second, _ = ExperimentRegistry._unit_identity("benchmark", item, "model", variant)
    assert first == second and len(hashes) == 6
    changed = variant.model_copy(update={"inference_configuration": {"confidence": .7}})
    assert ExperimentRegistry._unit_identity("benchmark", item, "model", changed)[0] != first


def test_public_asset_response_hides_local_root(tmp_path):
    settings = _settings(tmp_path); settings.prepare(); registry, dataset = _registered_dataset(tmp_path / "source")
    # Install the record into the app's own registry without exposing its path.
    app = create_app(settings); app.state.services.imported_assets.datasets.put(dataset.dataset_id, dataset.model_dump(mode="json"))
    with TestClient(app) as client:
        response = client.get("/api/v1/assets/datasets")
        samples = client.get(f"/api/v1/assets/datasets/{dataset.dataset_id}/samples?limit=2")
        thumbnail = client.get(samples.json()[0]["url"])
    assert response.status_code == 200 and "local_root" not in response.text and '"items"' not in response.text
    assert samples.status_code == 200 and len(samples.json()) == 2
    assert thumbnail.status_code == 200 and thumbnail.headers["content-type"] == "image/jpeg" and thumbnail.content.startswith(b"\xff\xd8")


def test_public_hub_api_surface_is_registered(tmp_path):
    app = create_app(_settings(tmp_path))
    paths = {route.path for route in app.routes}
    assert "/api/v1/public-hub/downloads" in paths
    assert "/api/v1/experiments/{experiment_id}/promote" in paths
    assert "/api/v1/recipes/execute" in paths


def test_promoted_recipe_is_pinned_versioned_and_auto_annotation_compatible(tmp_path):
    assets, dataset = _registered_dataset(tmp_path)
    benchmarks = BenchmarkRegistry(tmp_path / "benchmarks", assets)
    benchmark = benchmarks.create(BenchmarkCreate(name="fixed", dataset_id=dataset.dataset_id, requested_size=2))
    model_registry = ModelRegistry.load(Path(__file__).parents[2] / "config/models.yaml")
    weights = WeightStore(tmp_path / "weights", model_registry, 1024)
    weight = weights.save(io.BytesIO(b"inert onnx fixture"), "model.onnx", "onnx")
    workflow = {
        "version": "0.2", "name": "variant",
        "nodes": [
            {"id": "preview", "type": "tensor_preview", "config": {}},
            {"id": "model", "type": "model", "config": {"family": "onnx", "version": "generic", "size": "custom", "weight_id": weight.weight_id, "device": "cpu", "precision": "fp32"}},
            {"id": "source", "type": "image_source", "config": {"asset_id": "placeholder"}},
            {"id": "resize", "type": "resize", "config": {"width": 640, "height": 640}},
        ],
        "edges": [
            {"id": "e1", "source": "source", "source_handle": "image", "target": "resize", "target_handle": "image"},
            {"id": "e2", "source": "resize", "source_handle": "image", "target": "model", "target_handle": "input"},
            {"id": "e3", "source": "model", "source_handle": "output", "target": "preview", "target_handle": "result"},
        ],
    }
    experiment = InferenceExperiment(experiment_id="exp_recipe", name="Comparison", benchmark_id=benchmark.benchmark_id, variants=[RecipeVariant(variant_id="a", name="A", workflow=workflow)], status=ExperimentState.COMPLETED)
    recipes = RecipeRegistry(tmp_path / "recipes", benchmarks, weights)
    first = recipes.promote(experiment, PromoteRequest(variant_id="a"))
    second = recipes.promote(experiment, PromoteRequest(variant_id="a", recipe_id=first.recipe_id))
    assert first.recipe_version == 1 and second.recipe_version == 2
    assert recipes.get(first.recipe_id, 1).workflow_hash == first.workflow_hash
    assert {node["id"] for node in first.preprocessing_graph["nodes"]} == {"source", "resize"}
    assert {node["id"] for node in first.postprocessing_graph["nodes"]} == {"preview"}
    public = recipes.public(first)
    assert public["auto_annotation_recipe"]["weight_sha256"] == weight.sha256
    assert "workflow" not in public and len(recipes.list()) == 2


def test_detection_evaluation_computes_real_precision_recall_and_map():
    item = {"classes": ["helmet"], "ground_truth": [{"label": "helmet", "bbox": [0, 0, 10, 10], "polygons": []}]}
    result = {"result_type": "detections", "detections": [{"label": "helmet", "bbox": [0, 0, 10, 10], "confidence": .9}]}
    unit = ExperimentRegistry._evaluation(item, result)
    summary = ExperimentRegistry._summarize_evaluation([unit])
    assert summary == {"precision": 1.0, "recall": 1.0, "mAP50": 1.0, "mAP50-95": 1.0}
    assert ExperimentRegistry._evaluation({"classes": []}, result) == {}


def test_segmentation_evaluation_computes_mask_iou_and_dice():
    square = [1, 1, 8, 1, 8, 8, 1, 8]
    item = {"classes": ["object"], "width": 10, "height": 10, "ground_truth": [{"label": "object", "bbox": [1, 1, 8, 8], "polygons": [square]}]}
    result = {"result_type": "segmentation", "masks": [{"label": "object", "polygons": [square], "confidence": .95}]}
    summary = ExperimentRegistry._summarize_evaluation([ExperimentRegistry._evaluation(item, result)])
    assert summary["mask_IoU"] == 1.0 and summary["Dice"] == 1.0


def test_classification_evaluation_computes_accuracy_precision_recall_f1():
    rows = [
        ExperimentRegistry._evaluation({"classes": ["cat"]}, {"result_type": "classification", "predictions": [{"label": "cat"}]}),
        ExperimentRegistry._evaluation({"classes": ["dog"]}, {"result_type": "classification", "predictions": [{"label": "dog"}]}),
    ]
    summary = ExperimentRegistry._summarize_evaluation(rows)
    assert summary == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "F1": 1.0}


def test_credential_store_validates_persists_without_echo_and_removes(monkeypatch, tmp_path):
    provider = FakeProvider(); provider.name = "huggingface"; provider.display_name = "Hugging Face"
    store = ProviderCredentialStore(tmp_path / ".env", ExternalProviderRegistry([provider]))
    result = store.update("huggingface", ProviderCredentialUpdate(token="hf_test_value"))
    assert result == {"provider": "huggingface", "auth_state": "CONFIGURED", "configured": True}
    assert "hf_test_value" not in json.dumps(result) and "hf_test_value" not in json.dumps(store.status())
    assert "HF_TOKEN='hf_test_value'" in (tmp_path / ".env").read_text()
    assert oct((tmp_path / ".env").stat().st_mode & 0o777) == "0o600"
    removed = store.remove("huggingface")
    assert removed["configured"] is False and "HF_TOKEN=" not in (tmp_path / ".env").read_text()
    monkeypatch.delenv("HF_TOKEN", raising=False)
