from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from inference_studio.public_hub.schemas import utc_now


RUN_SCHEMA_VERSION = 1
CANONICAL_SCHEMA_VERSION = 1


class SourceType(str, Enum):
    IMAGE_FOLDER = "IMAGE_FOLDER"
    VIDEO = "VIDEO"


class SamplingMode(str, Enum):
    TARGET_FPS = "TARGET_FPS"
    INTERVAL_SECONDS = "INTERVAL_SECONDS"
    EVERY_N_FRAMES = "EVERY_N_FRAMES"


class RetentionMode(str, Enum):
    ALL_SAMPLED = "ALL_SAMPLED"
    ANY_PREDICTION = "ANY_PREDICTION"
    TOTAL_OBJECT_COUNT = "TOTAL_OBJECT_COUNT"
    CLASS_PRESENT = "CLASS_PRESENT"
    CLASS_COUNT = "CLASS_COUNT"


class ExportFormat(str, Enum):
    CANONICAL = "CANONICAL"
    COCO = "COCO"
    YOLO = "YOLO"


class AutoAnnotationRunState(str, Enum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    FINALIZING_MEDIA = "FINALIZING_MEDIA"
    EXPORTING_ANNOTATIONS = "EXPORTING_ANNOTATIONS"
    BUILDING_MANIFEST = "BUILDING_MANIFEST"
    VALIDATING_PACKAGE = "VALIDATING_PACKAGE"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class AutoAnnotationItemState(str, Enum):
    PENDING = "PENDING"
    SAMPLED = "SAMPLED"
    PROCESSING = "PROCESSING"
    INFERRED = "INFERRED"
    RETAINED = "RETAINED"
    EXPORTED = "EXPORTED"
    DISCARDED = "DISCARDED"
    FAILED = "FAILED"


class GeometryType(str, Enum):
    BBOX = "BBOX"
    POLYGON = "POLYGON"
    MASK = "MASK"
    KEYPOINTS = "KEYPOINTS"


class SourceConfig(BaseModel):
    path: str | None = None
    asset_id: str | None = None
    recursive: bool = True

    @model_validator(mode="after")
    def one_source(self):
        if bool(self.path) == bool(self.asset_id):
            raise ValueError("Provide exactly one source path or asset_id")
        return self


class FrameSamplingPolicy(BaseModel):
    mode: SamplingMode = SamplingMode.TARGET_FPS
    target_fps: float | None = Field(default=5.0, gt=0)
    interval_seconds: float | None = Field(default=None, gt=0)
    every_n_frames: int | None = Field(default=None, ge=1)
    start_seconds: float | None = Field(default=0, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_mode(self):
        required = {
            SamplingMode.TARGET_FPS: self.target_fps,
            SamplingMode.INTERVAL_SECONDS: self.interval_seconds,
            SamplingMode.EVERY_N_FRAMES: self.every_n_frames,
        }[self.mode]
        if required is None:
            raise ValueError(f"{self.mode.value} requires its sampling value")
        if self.end_seconds is not None and self.end_seconds <= (
            self.start_seconds or 0
        ):
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class FrameRetentionPolicy(BaseModel):
    mode: RetentionMode = RetentionMode.ALL_SAMPLED
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    min_total_objects: int | None = Field(default=None, ge=1)
    class_name: str | None = None
    min_class_count: int | None = Field(default=None, ge=1)
    min_area_px: float | None = Field(default=None, ge=0)
    min_save_interval_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_mode(self):
        if (
            self.mode == RetentionMode.TOTAL_OBJECT_COUNT
            and self.min_total_objects is None
        ):
            raise ValueError("TOTAL_OBJECT_COUNT requires min_total_objects")
        if (
            self.mode in {RetentionMode.CLASS_PRESENT, RetentionMode.CLASS_COUNT}
            and not (self.class_name or "").strip()
        ):
            raise ValueError(f"{self.mode.value} requires class_name")
        if self.mode == RetentionMode.CLASS_COUNT and self.min_class_count is None:
            raise ValueError("CLASS_COUNT requires min_class_count")
        return self


class RecipePin(BaseModel):
    recipe_id: str
    recipe_version: int
    recipe_hash: str
    workflow_id: str
    workflow_version: str
    workflow_hash: str
    model_family: str | None = None
    model_variant: str | None = None
    weight_id: str | None = None
    weight_sha256: str | None = None
    preprocessing_hash: str
    inference_hash: str
    postprocessing_hash: str
    device: str | None = None
    precision: str | None = None
    patch_size: int | None = None


class CanonicalGeometry(BaseModel):
    type: GeometryType
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    polygons: list[list[float]] = Field(default_factory=list)
    mask: dict[str, Any] | None = None
    keypoints: list[float] | None = None

    @model_validator(mode="after")
    def validate_geometry(self):
        if self.bbox is not None:
            if not all(isfinite(value) for value in self.bbox):
                raise ValueError("bbox coordinates must be finite")
            if self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]:
                raise ValueError("bbox must use XYXY order")
        if self.type == GeometryType.BBOX and self.bbox is None:
            raise ValueError("BBOX geometry requires bbox")
        if self.type == GeometryType.POLYGON:
            if not self.polygons:
                raise ValueError("POLYGON geometry requires polygons")
            if any(
                len(polygon) < 6
                or len(polygon) % 2
                or not all(isfinite(value) for value in polygon)
                for polygon in self.polygons
            ):
                raise ValueError(
                    "polygons require at least three finite coordinate pairs"
                )
        if self.type == GeometryType.MASK and self.mask is None:
            raise ValueError("MASK geometry requires mask data")
        if self.type == GeometryType.KEYPOINTS and not self.keypoints:
            raise ValueError("KEYPOINTS geometry requires keypoints")
        if self.keypoints and not all(isfinite(value) for value in self.keypoints):
            raise ValueError("keypoints must be finite")
        return self


class PredictionProvenance(BaseModel):
    recipe_id: str
    recipe_version: int
    recipe_hash: str
    workflow_id: str
    workflow_version: str
    workflow_hash: str
    model_family: str | None = None
    model_variant: str | None = None
    weight_id: str | None = None
    weight_sha256: str | None = None


class SourceProvenance(BaseModel):
    source_type: SourceType
    source_image: str | None = None
    source_sha256: str | None = None
    source_video_id: str | None = None
    source_video_name: str | None = None
    source_frame_index: int | None = None
    source_timestamp_seconds: float | None = None
    source_pts: int | None = None
    source_time_base: str | None = None


class CanonicalPrediction(BaseModel):
    schema_version: int = CANONICAL_SCHEMA_VERSION
    prediction_id: str
    source_item_id: str
    category_id: int = Field(ge=0)
    category_name: str = Field(min_length=1)
    confidence: float
    geometry: CanonicalGeometry
    attributes: dict[str, Any] = Field(default_factory=dict)
    provenance: PredictionProvenance
    source_provenance: SourceProvenance

    @model_validator(mode="after")
    def finite_confidence(self):
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between zero and one")
        return self


class AutoAnnotationItem(BaseModel):
    item_id: str
    run_id: str
    source_type: SourceType
    source_path_or_ref: str
    source_sha256: str | None = None
    relative_source_path: str | None = None
    source_video_id: str | None = None
    source_video_name: str | None = None
    frame_index: int | None = None
    pts: int | None = None
    time_base: str | None = None
    timestamp_seconds: float | None = None
    width: int = 0
    height: int = 0
    state: AutoAnnotationItemState = AutoAnnotationItemState.PENDING
    prediction_count: int = 0
    retained_reason: str | None = None
    discarded_reason: str | None = None
    output_image_relative_path: str | None = None
    predictions: list[CanonicalPrediction] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error_stage: str | None = None
    error: str | None = None
    updated_at: str = Field(default_factory=utc_now)


class AutoAnnotationRunCreate(BaseModel):
    source_type: SourceType
    source_config: SourceConfig
    recipe_id: str
    recipe_version: int = Field(ge=1)
    sampling_policy: FrameSamplingPolicy | None = None
    retention_policy: FrameRetentionPolicy = Field(default_factory=FrameRetentionPolicy)
    requested_exports: list[ExportFormat] = Field(
        default_factory=lambda: list(ExportFormat), min_length=1
    )
    jpeg_quality: int = Field(default=92, ge=50, le=100)

    @model_validator(mode="after")
    def validate_source_policy(self):
        if self.source_type == SourceType.VIDEO and self.sampling_policy is None:
            raise ValueError("Video runs require a sampling policy")
        if self.source_type == SourceType.IMAGE_FOLDER and self.source_config.asset_id:
            raise ValueError("Image-folder runs require a configured local folder")
        if len(set(self.requested_exports)) != len(self.requested_exports):
            raise ValueError("Export formats must be unique")
        return self


class AutoAnnotationRun(BaseModel):
    schema_version: int = RUN_SCHEMA_VERSION
    feature_version: str = "Auto Annotation v0.1"
    run_id: str
    source_type: SourceType
    source_config: SourceConfig
    recipe_id: str
    recipe_version: int
    recipe_hash: str | None = None
    recipe_pin: RecipePin | None = None
    output_root: str
    sampling_policy: FrameSamplingPolicy | None = None
    retention_policy: FrameRetentionPolicy
    requested_exports: list[ExportFormat]
    jpeg_quality: int = 92
    state: AutoAnnotationRunState = AutoAnnotationRunState.CREATED
    source_count: int = 0
    sampled_count: int = 0
    inferred_count: int = 0
    retained_count: int = 0
    exported_count: int = 0
    failed_count: int = 0
    prediction_count: int = 0
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    package_validation: dict[str, Any] = Field(default_factory=dict)
    last_item: dict[str, Any] = Field(default_factory=dict)
    ready_for_upload: bool = False
    cancel_requested: bool = False
    recoverable: bool = True
    created_at: str = Field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = Field(default_factory=utc_now)
    error: str | None = None


class PreflightIssue(BaseModel):
    code: str
    message: str
    field: str | None = None


class RecipePreflight(BaseModel):
    status: Literal["READY", "NOT_READY"]
    recipe: dict[str, Any] | None = None
    pin: RecipePin | None = None
    errors: list[PreflightIssue] = Field(default_factory=list)


class SourceInspectRequest(BaseModel):
    source_type: SourceType
    source_config: SourceConfig


class SamplingPreviewRequest(SourceInspectRequest):
    sampling_policy: FrameSamplingPolicy
    limit: int = Field(default=12, ge=1, le=24)


class InferencePreviewRequest(SourceInspectRequest):
    recipe_id: str
    recipe_version: int = Field(ge=1)
    sampling_policy: FrameSamplingPolicy | None = None
    retention_policy: FrameRetentionPolicy = Field(default_factory=FrameRetentionPolicy)
    limit: int = Field(default=10, ge=1, le=12)
