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
from inference_studio.public_hub.base import DownloadedResource, ExternalProvider, run_with_progress
from inference_studio.public_hub.benchmarks import BenchmarkRegistry
from inference_studio.public_hub.downloads import DownloadManager
from inference_studio.public_hub.credentials import ProviderCredentialStore
from inference_studio.public_hub.experiments import ExperimentRegistry
from inference_studio.public_hub.inspectors import DatasetInspector, ModelCompatibilityInspector
from inference_studio.public_hub.integrity import IntegrityError, safe_extract_archive
from inference_studio.public_hub.registry import ExternalProviderRegistry
from inference_studio.public_hub.recipes import RecipeRegistry
from inference_studio.public_hub.schemas import (
    AuthState, BenchmarkCreate, CanvasRecipePromoteRequest, DownloadState, ExternalResource, ExternalResourceRef,
    ExperimentState, InferenceExperiment, PromoteRequest, ProviderCapabilities, ProviderCredentialUpdate, RecipeVariant,
    ResourceDownloadRequest, ResourceKind, SearchPage,
)


def test_checkpoint_family_suggestion_distinguishes_yolo_segmentation():
    families = ["yolo", "yolo-seg", "rtdetr"]
    assert api_routes._suggest_family("weights/yolo11n-seg.pt", families) == "yolo-seg"
    assert api_routes._suggest_family("weights/yolo11n.pt", families) == "yolo"


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


def test_model_inspector_recognizes_rtdetr_from_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "weights" / "RT-DETR" / "best.pt"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"never loaded during inspection")
    result = ModelCompatibilityInspector().inspect(tmp_path)
    assert result.state.value == "READY" and result.candidate_adapter == "rtdetr"
    assert result.artifact_file == "weights/RT-DETR/best.pt"


def test_model_inspector_recognizes_official_rtdetrv2_pth_from_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "weights" / "RT-DETRv2" / "best.pth"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"never loaded during inspection")
    result = ModelCompatibilityInspector().inspect(tmp_path)
    assert result.state.value == "READY" and result.candidate_adapter == "rtdetrv2"
    assert result.artifact_file == "weights/RT-DETRv2/best.pth"


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


class SlowSdkProvider(FakeProvider):
    """Mimics a blocking SDK download (huggingface_hub/kaggle): no byte callback, files land over time."""

    CHUNK, CHUNKS = 64 * 1024, 6

    def __init__(self, manager_ref: dict, delay: float = 0.15):
        super().__init__()
        self.manager_ref, self.delay = manager_ref, delay
        self.seen: list[tuple[int, int | None]] = []
        self.states_during_transfer: list[DownloadState] = []

    def details(self, ref):
        return ExternalResource(ref=ref, title="Resource", resolved_revision="abc123", license="MIT", size_bytes=self.CHUNK * self.CHUNKS)

    def download(self, ref, destination, requested_format, progress, cancelled):
        import time
        destination.mkdir(parents=True)
        identity = destination.parent.name

        def work():
            for index in range(self.CHUNKS):
                if cancelled():
                    return
                (destination / f"part{index}.bin").write_bytes(b"x" * self.CHUNK)
                self.states_during_transfer.append(self.manager_ref["manager"].get(identity).state)
                time.sleep(self.delay)

        def recording(done, total):
            self.seen.append((done, total))
            progress(done, total)

        run_with_progress(work, destination, self.CHUNK * self.CHUNKS, recording, cancelled, interval=0.02)
        if cancelled():
            raise InterruptedError("Download cancelled")
        return DownloadedResource(destination, "abc123")


@pytest.mark.asyncio
async def test_blocking_sdk_download_reports_live_progress_and_downloading_state(tmp_path):
    settings = _settings(tmp_path); settings.prepare(); ref: dict = {}
    provider = SlowSdkProvider(ref, delay=0.05)
    manager = ref["manager"] = DownloadManager(settings, ExternalProviderRegistry([provider]), ImportedAssetRegistry(settings.public_hub_root / "metadata/assets"))
    job = manager.create(ResourceDownloadRequest(resource=ExternalResourceRef(provider="fake", kind=ResourceKind.DATASET, resource_id="owner/slow")))
    await manager._tasks[job.download_id]
    total = SlowSdkProvider.CHUNK * SlowSdkProvider.CHUNKS
    assert provider.states_during_transfer and set(provider.states_during_transfer) == {DownloadState.DOWNLOADING}
    partial = [done for done, expected in provider.seen if 0 < done < total and expected == total]
    assert partial, provider.seen  # progress moved before the transfer finished
    assert [done for done, _ in provider.seen] == sorted(done for done, _ in provider.seen)


@pytest.mark.asyncio
async def test_cancel_interrupts_a_blocking_sdk_download_promptly(tmp_path):
    import asyncio, time
    settings = _settings(tmp_path); settings.prepare(); ref: dict = {}
    provider = SlowSdkProvider(ref, delay=0.3)  # whole transfer would take about 1.8 s
    manager = ref["manager"] = DownloadManager(settings, ExternalProviderRegistry([provider]), ImportedAssetRegistry(settings.public_hub_root / "metadata/assets"))
    job = manager.create(ResourceDownloadRequest(resource=ExternalResourceRef(provider="fake", kind=ResourceKind.DATASET, resource_id="owner/cancel")))
    while manager.get(job.download_id).state != DownloadState.DOWNLOADING:
        await asyncio.sleep(0.01)
    started = time.monotonic()
    manager.cancel(job.download_id)
    await manager._tasks[job.download_id]
    final = manager.get(job.download_id)
    assert final.state == DownloadState.CANCELLED and time.monotonic() - started < 0.5
    assert not (manager.staging / job.download_id).exists()


@pytest.mark.asyncio
async def test_recover_honours_a_cancel_requested_before_restart(tmp_path):
    settings = _settings(tmp_path); settings.prepare()
    manager = DownloadManager(settings, ExternalProviderRegistry([FakeProvider()]), ImportedAssetRegistry(settings.public_hub_root / "metadata/assets"))
    from inference_studio.public_hub.schemas import ResourceDownloadJob
    stuck = ResourceDownloadJob(download_id="dl_" + "a" * 32, resource=ExternalResourceRef(provider="fake", kind=ResourceKind.DATASET, resource_id="owner/stuck"), state=DownloadState.RESOLVING, cancel_requested=True)
    manager._save(stuck)
    (manager.staging / stuck.download_id).mkdir()
    await manager.recover()
    assert manager.get(stuck.download_id).state == DownloadState.CANCELLED
    assert stuck.download_id not in manager._tasks and not (manager.staging / stuck.download_id).exists()


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
    assert "/api/v1/recipes/from-canvas" in paths
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


def test_canvas_workflow_can_be_frozen_directly_for_auto_annotation(tmp_path):
    assets, _dataset = _registered_dataset(tmp_path)
    benchmarks = BenchmarkRegistry(tmp_path / "benchmarks", assets)
    model_registry = ModelRegistry.load(Path(__file__).parents[2] / "config/models.yaml")
    weights = WeightStore(tmp_path / "weights", model_registry, 1024)
    weight = weights.save(io.BytesIO(b"inert onnx fixture"), "model.onnx", "onnx")
    workflow = {
        "version": "0.2", "name": "Canvas workspace",
        "nodes": [
            {"id": "source", "type": "image_source", "config": {}},
            {"id": "model", "type": "model", "config": {"family": "onnx", "version": "generic", "size": "custom", "weight_id": weight.weight_id, "device": "cpu", "precision": "fp32"}},
            {"id": "preview", "type": "tensor_preview", "config": {}},
        ],
        "edges": [
            {"id": "e1", "source": "source", "source_handle": "image", "target": "model", "target_handle": "input"},
            {"id": "e2", "source": "model", "source_handle": "output", "target": "preview", "target_handle": "result"},
        ],
    }
    recipes = RecipeRegistry(tmp_path / "recipes", benchmarks, weights)
    recipe = recipes.promote_canvas(CanvasRecipePromoteRequest(workflow=workflow, name="Direct Canvas"))
    assert recipe.lifecycle.value == "PROMOTED"
    assert recipe.promotion_source == "CANVAS"
    assert recipe.benchmark_id is None and recipe.source_experiment_id is None
    assert recipe.weight_sha256 == weight.sha256
    assert recipes.get(recipe.recipe_id, 1).workflow_hash == recipe.workflow_hash


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


def test_credential_store_keeps_vllm_bearer_token_server_side(monkeypatch, tmp_path):
    provider = FakeProvider(); provider.name = "huggingface"; provider.display_name = "Hugging Face"
    store = ProviderCredentialStore(tmp_path / ".env", ExternalProviderRegistry([provider]))
    result = store.update("vllm", ProviderCredentialUpdate(api_key="private-vllm-token"))
    assert result == {"provider": "vllm", "auth_state": "CONFIGURED", "configured": True}
    assert "private-vllm-token" not in json.dumps(store.status())
    assert "VLLM_API_KEY='private-vllm-token'" in (tmp_path / ".env").read_text()
    assert store.remove("vllm")["configured"] is False
    monkeypatch.delenv("VLLM_API_KEY", raising=False)


def test_dataset_sample_import_creates_a_reusable_canvas_image_asset(tmp_path):
    settings = _settings(tmp_path); settings.prepare(); _, dataset = _registered_dataset(tmp_path / "source")
    app = create_app(settings); app.state.services.imported_assets.datasets.put(dataset.dataset_id, dataset.model_dump(mode="json"))
    item = dataset.inspection.items[0]
    with TestClient(app) as client:
        first = client.post(f"/api/v1/assets/datasets/{dataset.dataset_id}/samples/{item['item_id']}/import")
        again = client.post(f"/api/v1/assets/datasets/{dataset.dataset_id}/samples/{item['item_id']}/import")
        missing = client.post(f"/api/v1/assets/datasets/{dataset.dataset_id}/samples/nope/import")
        unknown = client.post(f"/api/v1/assets/datasets/ds_unknown/samples/{item['item_id']}/import")
        asset = first.json()
        workflow = {"version": "0.2", "name": "sample", "nodes": [
            {"id": "src", "type": "image_source", "position": {"x": 0, "y": 0}, "config": {"asset_id": asset["asset_id"]}},
            {"id": "view", "type": "image_preview", "position": {"x": 1, "y": 0}, "config": {}}],
            "edges": [{"id": "e", "source": "src", "source_handle": "image", "target": "view", "target_handle": "image"}]}
        validation = client.post("/api/v1/workflows/validate", json=workflow).json()
    assert first.status_code == 201 and asset["kind"] == "image" and asset["asset_id"].startswith("ast_")
    assert (asset["width"], asset["height"]) == (item["width"], item["height"])  # full resolution, not the thumbnail
    assert "path" not in asset and "local_root" not in first.text
    assert again.json()["asset_id"] == asset["asset_id"]  # identical bytes reuse one asset
    assert missing.status_code == 404 and unknown.status_code == 404
    assert validation["valid"], validation


def test_ambiguous_model_artifact_files_can_be_listed_and_registered(tmp_path):
    from inference_studio.public_hub.schemas import InspectionState, ModelInspection
    settings = _settings(tmp_path); settings.prepare()
    root = tmp_path / "bmd"  # mirrors iisc-aim/BMD-45: several checkpoints, only some with a built-in adapter
    for relative in ("weights/YOLOv12-S/best.pt", "weights/RT-DETR/rtdetr-l.pt", "weights/RT-DETRv2/best.pth", "weights/RF-DETR/checkpoint_best_total.pth", "weights/D-FINE/best_stg1.pth", ".cache/huggingface/stale.pt", "README.md"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True); (root / relative).write_bytes(b"checkpoint-" + relative.encode())
    app = create_app(settings)
    resource = ExternalResource(ref=ExternalResourceRef(provider="huggingface", kind=ResourceKind.MODEL, resource_id="iisc-aim/BMD-45", revision="1"), title="BMD-45", resolved_revision="1")
    artifact = app.state.services.imported_assets.register_model(root, resource, ModelInspection(state=InspectionState.AMBIGUOUS, explanation="Multiple incompatible artifact formats were found"))
    base = f"/api/v1/assets/models/{artifact.model_artifact_id}"
    with TestClient(app) as client:
        listing = client.get(f"{base}/artifacts").json()
        files = {item["relative_path"]: item for item in listing["files"]}
        registered = client.post(f"{base}/artifacts/register", json={"relative_path": "weights/YOLOv12-S/best.pt", "family": "yolo"})
        repeated = client.post(f"{base}/artifacts/register", json={"relative_path": "weights/YOLOv12-S/best.pt", "family": "yolo"})
        traversal = client.post(f"{base}/artifacts/register", json={"relative_path": "../outside.pt", "family": "yolo"})
        wrong_extension = client.post(f"{base}/artifacts/register", json={"relative_path": "weights/D-FINE/best_stg1.pth", "family": "onnx"})
        unknown_family = client.post(f"{base}/artifacts/register", json={"relative_path": "weights/D-FINE/best_stg1.pth", "family": "nope"})
        weights = client.get("/api/v1/weights").json()
        after = client.get(base).json(); relisted = client.get(f"{base}/artifacts").json()
    assert set(files) == {"weights/YOLOv12-S/best.pt", "weights/RT-DETR/rtdetr-l.pt", "weights/RT-DETRv2/best.pth", "weights/RF-DETR/checkpoint_best_total.pth", "weights/D-FINE/best_stg1.pth"}  # .cache and non-weights skipped
    assert files["weights/YOLOv12-S/best.pt"]["suggested_family"] == "yolo" and "yolo" in files["weights/YOLOv12-S/best.pt"]["families"]
    assert files["weights/RT-DETR/rtdetr-l.pt"]["suggested_family"] == "rtdetr"
    assert files["weights/RT-DETRv2/best.pth"]["suggested_family"] == "rtdetrv2"
    assert "rtdetrv2" in files["weights/RT-DETRv2/best.pth"]["families"]
    assert files["weights/RF-DETR/checkpoint_best_total.pth"]["suggested_family"] == "rfdetr"
    assert files["weights/D-FINE/best_stg1.pth"]["suggested_family"] is None
    assert registered.status_code == 201 and repeated.json()["weight_id"] == registered.json()["weight_id"]
    assert any(item["weight_id"] == registered.json()["weight_id"] and item["family"] == "yolo" for item in weights)
    assert after["weight_id"] == registered.json()["weight_id"] and len(after["registered_weights"]) == 1 and "local_root" not in after
    assert next(item for item in relisted["files"] if item["relative_path"] == "weights/YOLOv12-S/best.pt")["weight_id"] == registered.json()["weight_id"]
    assert traversal.status_code == 404 and wrong_extension.status_code == 422 and unknown_family.status_code == 422


def test_path_detection_repairs_legacy_rfdetr_registration_for_rtdetrv2(tmp_path):
    from inference_studio.public_hub.schemas import InspectionState, ModelInspection

    settings = _settings(tmp_path); settings.prepare()
    root = tmp_path / "bmd"
    checkpoint = root / "weights" / "RT-DETRv2" / "best.pth"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"native-rtdetrv2-checkpoint")
    app = create_app(settings)
    resource = ExternalResource(ref=ExternalResourceRef(provider="huggingface", kind=ResourceKind.MODEL, resource_id="iisc-aim/BMD-45", revision="1"), title="BMD-45", resolved_revision="1")
    artifact = app.state.services.imported_assets.register_model(root, resource, ModelInspection(state=InspectionState.AMBIGUOUS, explanation="test"))
    base = f"/api/v1/assets/models/{artifact.model_artifact_id}/artifacts"
    with TestClient(app) as client:
        legacy = client.post(f"{base}/register", json={"relative_path": "weights/RT-DETRv2/best.pth", "family": "rfdetr"})
        repaired = client.get(base).json()["files"][0]
        correct = client.post(f"{base}/register", json={"relative_path": "weights/RT-DETRv2/best.pth", "family": "rtdetrv2"})
        relisted = client.get(base).json()["files"][0]
    assert legacy.status_code == 201
    assert repaired["suggested_family"] == "rtdetrv2" and repaired["weight_id"] is None
    assert correct.status_code == 201
    assert relisted["suggested_family"] == "rtdetrv2" and relisted["weight_id"] == correct.json()["weight_id"]
