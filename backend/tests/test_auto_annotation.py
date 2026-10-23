from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference_studio.auto_annotation.canonical import canonical_predictions
from inference_studio.auto_annotation.package import PackageBuilder
from inference_studio.auto_annotation.manager import AutoAnnotationManager
from inference_studio.auto_annotation.repository import AutoAnnotationRunRepository
from inference_studio.auto_annotation.retention import RetentionEvaluator
from inference_studio.auto_annotation.sampling import FrameSampler
from inference_studio.auto_annotation.schemas import (
    AutoAnnotationItem,
    AutoAnnotationItemState,
    AutoAnnotationRun,
    AutoAnnotationRunState,
    CanonicalGeometry,
    CanonicalPrediction,
    ExportFormat,
    FrameRetentionPolicy,
    FrameSamplingPolicy,
    GeometryType,
    PredictionProvenance,
    RecipePin,
    RetentionMode,
    SamplingMode,
    SourceConfig,
    SourceProvenance,
    SourceType,
    AutoAnnotationRunCreate,
    RecipePreflight,
)
from inference_studio.auto_annotation.sources import SourceAccessError, SourceService
from inference_studio.assets import AssetStore
from inference_studio.config import Settings
from inference_studio.main import create_app


def pin() -> RecipePin:
    return RecipePin(
        recipe_id="rcp_test",
        recipe_version=2,
        recipe_hash="a" * 64,
        workflow_id="wf_test",
        workflow_version="0.2",
        workflow_hash="b" * 64,
        model_family="yolo",
        model_variant="v8:n",
        weight_id="wgt_test",
        weight_sha256="c" * 64,
        preprocessing_hash="d" * 64,
        inference_hash="e" * 64,
        postprocessing_hash="f" * 64,
        device="cpu",
        precision="fp32",
    )


def provenance(source_type: SourceType = SourceType.IMAGE_FOLDER) -> SourceProvenance:
    return SourceProvenance(source_type=source_type, source_image="nested/image.jpg")


def prediction(
    item_id: str = "aai_test_item", name: str = "person", confidence: float = 0.8
) -> CanonicalPrediction:
    return CanonicalPrediction(
        prediction_id="pred_1",
        source_item_id=item_id,
        category_id=0,
        category_name=name,
        confidence=confidence,
        geometry=CanonicalGeometry(type=GeometryType.BBOX, bbox=[1, 2, 11, 12]),
        provenance=PredictionProvenance(
            **pin().model_dump(
                include={
                    "recipe_id",
                    "recipe_version",
                    "recipe_hash",
                    "workflow_id",
                    "workflow_version",
                    "workflow_hash",
                    "model_family",
                    "model_variant",
                    "weight_id",
                    "weight_sha256",
                }
            )
        ),
        source_provenance=provenance(),
    )


def test_timestamp_sampling_is_deterministic_and_range_bounded():
    policy = FrameSamplingPolicy(
        mode=SamplingMode.TARGET_FPS, target_fps=2, start_seconds=0.5, end_seconds=1.5
    )
    first = FrameSampler(policy, 10)
    second = FrameSampler(policy, 10)
    selected = [index for index in range(21) if first.select(index, index / 10)]
    assert selected == [5, 10, 15]
    assert selected == [
        index for index in range(21) if second.select(index, index / 10)
    ]
    assert FrameSampler.estimated_count(policy, 10, 21, 2.1) == 3


def test_every_n_sampling_anchors_to_start_frame():
    sampler = FrameSampler(
        FrameSamplingPolicy(
            mode=SamplingMode.EVERY_N_FRAMES,
            target_fps=None,
            every_n_frames=3,
            start_seconds=0.2,
        ),
        10,
    )
    assert [index for index in range(12) if sampler.select(index, index / 10)] == [
        2,
        5,
        8,
        11,
    ]


def test_retention_applies_filters_and_global_temporal_spacing():
    policy = FrameRetentionPolicy(
        mode=RetentionMode.CLASS_COUNT,
        class_name="person",
        min_class_count=1,
        min_confidence=0.5,
        min_area_px=50,
        min_save_interval_seconds=2,
    )
    evaluator = RetentionEvaluator(policy)
    values = [prediction(confidence=0.8), prediction(name="car", confidence=0.9)]
    assert evaluator.evaluate(values, 1).retained
    spaced = evaluator.evaluate(values, 2)
    assert (
        not spaced.retained
        and spaced.prediction_rule_passed
        and not spaced.temporal_spacing_passed
    )
    assert evaluator.evaluate(values, 3).retained


def test_canonical_detection_preserves_recipe_and_source_provenance():
    values = canonical_predictions(
        {
            "result_type": "detections",
            "detections": [
                {
                    "class_id": 4,
                    "label": "helmet",
                    "confidence": 0.75,
                    "bbox": [2, 3, 9, 12],
                }
            ],
        },
        "aai_item",
        pin(),
        provenance(),
    )
    assert len(values) == 1
    assert values[0].geometry.bbox == [2, 3, 9, 12]
    assert values[0].provenance.recipe_version == 2
    assert values[0].source_provenance.source_image == "nested/image.jpg"


def test_canonical_segmentation_preserves_polygon_not_bitmap():
    values = canonical_predictions(
        {
            "result_type": "segmentation",
            "masks": [
                {
                    "class_id": 1,
                    "label": "part",
                    "confidence": 0.9,
                    "bbox": [0, 0, 10, 10],
                    "area": 50,
                    "polygons": [[0, 0, 10, 0, 10, 10]],
                }
            ],
        },
        "aai_item",
        pin(),
        provenance(),
    )
    assert values[0].geometry.type == GeometryType.POLYGON
    assert values[0].geometry.polygons == [[0, 0, 10, 0, 10, 10]]


def test_source_service_enforces_approved_roots_and_inventory(tmp_path):
    approved = tmp_path / "approved"
    approved.mkdir()
    assert cv2.imwrite(str(approved / "one.jpg"), np.zeros((8, 12, 3), np.uint8))
    service = SourceService(
        AssetStore(tmp_path / "assets", 1_000_000), [approved], tmp_path / "previews"
    )
    inspected = service.inspect(
        SourceType.IMAGE_FOLDER, SourceConfig(path=str(approved))
    )
    assert inspected["source_count"] == 1 and inspected["readable_count"] == 1
    original_snapshot = inspected["snapshot_sha256"]
    assert cv2.imwrite(str(approved / "one.jpg"), np.ones((8, 12, 3), np.uint8))
    changed = service.inspect(SourceType.IMAGE_FOLDER, SourceConfig(path=str(approved)))
    assert changed["snapshot_sha256"] != original_snapshot
    with pytest.raises(SourceAccessError, match="VIS_AUTO_ANNOTATION_INPUT_ROOTS"):
        service.inspect(SourceType.IMAGE_FOLDER, SourceConfig(path=str(tmp_path)))
    outside = tmp_path / "outside.jpg"
    assert cv2.imwrite(str(outside), np.zeros((8, 12, 3), np.uint8))
    (approved / "escape.jpg").symlink_to(outside)
    with pytest.raises(SourceAccessError, match="escapes its approved root"):
        service.inspect(SourceType.IMAGE_FOLDER, SourceConfig(path=str(approved)))


def _package(tmp_path: Path):
    repository = AutoAnnotationRunRepository(tmp_path / "state")
    root = tmp_path / "packages" / "AA-aa_test"
    (root / "images").mkdir(parents=True)
    run = AutoAnnotationRun(
        run_id="aa_test",
        source_type=SourceType.IMAGE_FOLDER,
        source_config=SourceConfig(path=str(tmp_path)),
        recipe_id="rcp_test",
        recipe_version=2,
        recipe_hash="a" * 64,
        recipe_pin=pin(),
        output_root=str(root),
        retention_policy=FrameRetentionPolicy(),
        requested_exports=list(ExportFormat),
        state=AutoAnnotationRunState.COMPLETED,
        ready_for_upload=True,
    )
    repository.create(run)
    relative = "images/aai_test_item.jpg"
    assert cv2.imwrite(str(root / relative), np.zeros((20, 30, 3), np.uint8))
    item = AutoAnnotationItem(
        item_id="aai_test_item",
        run_id=run.run_id,
        source_type=SourceType.IMAGE_FOLDER,
        source_path_or_ref=str(tmp_path / "secret" / "image.jpg"),
        relative_source_path="image.jpg",
        width=30,
        height=20,
        state=AutoAnnotationItemState.EXPORTED,
        prediction_count=1,
        output_image_relative_path=relative,
        predictions=[prediction()],
    )
    repository.record_item(item)
    return run, repository, PackageBuilder(repository), root


def test_package_exports_manifest_after_retained_media_and_validates(tmp_path):
    run, _repository, package, root = _package(tmp_path)
    exports = package.export_annotations(run)
    manifest_sha = package.build_manifest(run)
    package.write_metadata(run, exports, manifest_sha)
    assert package.validate(run) == {"valid": True, "errors": [], "exported_items": 1}
    manifest = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    assert [row.get("image") for row in manifest[1:]] == ["images/aai_test_item.jpg"]
    assert str(tmp_path) not in (root / "run.json").read_text()
    assert str(tmp_path) not in (root / "source.json").read_text()


def test_coco_and_yolo_exports_are_stable_and_normalized(tmp_path):
    run, _repository, package, root = _package(tmp_path)
    package.export_annotations(run)
    coco = json.loads((root / "annotations" / "coco.json").read_text())
    assert coco["categories"] == [
        {"id": 1, "name": "person", "supercategory": "object"}
    ]
    assert coco["annotations"][0]["bbox"] == [1.0, 2.0, 10.0, 10.0]
    values = (
        (root / "annotations" / "yolo" / "labels" / "aai_test_item.txt")
        .read_text()
        .split()
    )
    assert values[0] == "0" and all(0 <= float(value) <= 1 for value in values[1:])


def test_package_validation_detects_tampering(tmp_path):
    run, _repository, package, root = _package(tmp_path)
    exports = package.export_annotations(run)
    digest = package.build_manifest(run)
    package.write_metadata(run, exports, digest)
    (root / "images" / "aai_test_item.jpg").write_bytes(b"broken")
    result = package.validate(run)
    assert not result["valid"] and any(
        "Unreadable image" in error or "Checksum mismatch" in error
        for error in result["errors"]
    )


def test_auto_annotation_api_exposes_preflight_without_selecting_latest(tmp_path):
    root = Path(__file__).parents[2]
    settings = Settings(
        project_root=root,
        workspace_dir=tmp_path / "workspace",
        weights_dir=tmp_path / "weights",
        workflows_dir=tmp_path / "workflows",
        benchmarks_dir=tmp_path / "benchmarks",
        models_file=root / "config" / "models.yaml",
        auto_annotation_input_roots=[tmp_path],
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/auto-annotation/recipes/rcp_missing/versions/7/preflight"
        )
        runs = client.get("/api/v1/auto-annotation/runs")
    assert response.status_code == 200 and response.json()["status"] == "NOT_READY"
    assert "version 7" in response.json()["errors"][0]["message"]
    assert runs.status_code == 200 and runs.json() == []


@pytest.mark.asyncio
async def test_image_folder_run_completes_through_durable_package(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    assert cv2.imwrite(str(source / "one.jpg"), np.zeros((20, 30, 3), np.uint8))
    assert cv2.imwrite(str(source / "two.jpg"), np.ones((20, 30, 3), np.uint8))

    class Recipes:
        @staticmethod
        def get(_identity, _version):
            return object()

    class Executor:
        @staticmethod
        def preflight(_identity, _version):
            return RecipePreflight(status="READY", pin=pin(), recipe={})

        @staticmethod
        async def execute(_recipe, _image, _source_name, _source_id, **_options):
            return {
                "result": {
                    "result_type": "detections",
                    "detections": [
                        {
                            "class_id": 0,
                            "label": "part",
                            "confidence": 0.9,
                            "bbox": [1, 2, 11, 12],
                        }
                    ],
                },
                "metrics": {"inference_ms": 1.0},
            }

    assets = AssetStore(tmp_path / "assets", 1_000_000)
    manager = AutoAnnotationManager(
        tmp_path / "state",
        tmp_path / "packages",
        SourceService(assets, [source], tmp_path / "previews"),
        Recipes(),
        Executor(),  # type: ignore[arg-type]
    )
    run = manager.create(
        AutoAnnotationRunCreate(
            source_type=SourceType.IMAGE_FOLDER,
            source_config=SourceConfig(path=str(source)),
            recipe_id="rcp_test",
            recipe_version=2,
        )
    )
    await manager._tasks[run.run_id]
    completed = manager.get(run.run_id)
    assert completed.state == AutoAnnotationRunState.COMPLETED
    assert (
        completed.sampled_count
        == completed.inferred_count
        == completed.exported_count
        == 2
    )
    assert completed.prediction_count == 2 and completed.ready_for_upload
    assert manager.packages.validate(completed)["valid"]
    listing = manager.review_items(run.run_id, limit=1)
    assert listing["total"] == 2 and len(listing["items"]) == 1
    reviewed = manager.review_item(run.run_id, listing["items"][0]["item_id"])
    assert reviewed["prediction_count"] == 1
    assert reviewed["predictions"][0]["category_name"] == "part"
    assert reviewed["image_url"].endswith("/image")
    assert manager.review_image(run.run_id, reviewed["item_id"]).is_file()
    assert manager.review_items(
        run.run_id, prediction_filter="WITHOUT_PREDICTIONS"
    )["total"] == 0
