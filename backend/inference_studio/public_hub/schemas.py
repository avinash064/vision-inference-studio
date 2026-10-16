from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResourceKind(str, Enum):
    DATASET = "dataset"
    MODEL = "model"


class AuthState(str, Enum):
    CONFIGURED = "CONFIGURED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    AUTH_FAILED = "AUTH_FAILED"


class ProviderCapabilities(BaseModel):
    search_datasets: bool = False
    download_datasets: bool = False
    search_models: bool = False
    download_models: bool = False
    revisions: bool = False
    formats: bool = False


class ProviderCredentialUpdate(BaseModel):
    token: SecretStr | None = None
    api_token: SecretStr | None = None
    api_key: SecretStr | None = None
    username: str | None = Field(default=None, max_length=256)
    test_before_save: bool = Field(default=True, alias="validate")


class ExternalResourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    kind: ResourceKind
    resource_id: str = Field(min_length=1, max_length=512)
    version: str | None = Field(default=None, max_length=256)
    revision: str | None = Field(default=None, max_length=256)


class ExternalFile(BaseModel):
    name: str
    size_bytes: int | None = None
    checksum: str | None = None


class ExternalResource(BaseModel):
    ref: ExternalResourceRef
    title: str
    description: str = ""
    owner: str | None = None
    task: str | None = None
    architecture: str | None = None
    tags: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=list)
    files: list[ExternalFile] = Field(default_factory=list)
    image_count: int | None = None
    size_bytes: int | None = None
    license: str | None = None
    source_url: str | None = None
    preview_url: str | None = None
    resolved_revision: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchPage(BaseModel):
    items: list[ExternalResource]
    page: int
    page_size: int
    has_more: bool


class DownloadState(str, Enum):
    QUEUED = "QUEUED"
    RESOLVING = "RESOLVING"
    DOWNLOADING = "DOWNLOADING"
    VERIFYING = "VERIFYING"
    EXTRACTING = "EXTRACTING"
    INSPECTING = "INSPECTING"
    REGISTERING = "REGISTERING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class ResourceDownloadRequest(BaseModel):
    resource: ExternalResourceRef
    format: str | None = Field(default=None, max_length=64)


class ResourceDownloadJob(BaseModel):
    download_id: str
    resource: ExternalResourceRef
    requested_format: str | None = None
    state: DownloadState = DownloadState.QUEUED
    progress: float = 0
    bytes_downloaded: int = 0
    total_bytes: int | None = None
    speed_bytes_per_second: float | None = None
    resolved_revision: str | None = None
    checksum: str | None = None
    asset_id: str | None = None
    error: str | None = None
    retry_count: int = 0
    cancel_requested: bool = False
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    local_path: str | None = None


class InspectionState(str, Enum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    ADAPTER_REQUIRED = "ADAPTER_REQUIRED"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID = "INVALID"


class DatasetInspection(BaseModel):
    state: InspectionState
    format: str = "unknown"
    task: str = "unknown"
    image_count: int = 0
    valid_image_count: int = 0
    annotation_count: int = 0
    classes: list[str] = Field(default_factory=list)
    splits: dict[str, int] = Field(default_factory=dict)
    missing_labels: int = 0
    broken_files: list[str] = Field(default_factory=list)
    duplicate_images: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""


class ModelInspection(BaseModel):
    state: InspectionState
    framework: str | None = None
    architecture: str | None = None
    candidate_adapter: str | None = None
    artifact_file: str | None = None
    explanation: str
    ignored_python_files: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    provider: str
    resource_id: str
    requested_version: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    source_url: str | None = None
    owner: str | None = None
    license: str | None = None


class DatasetAsset(BaseModel):
    dataset_id: str
    display_name: str
    provenance: Provenance
    local_root: str
    format: str
    task: str
    status: InspectionState
    image_count: int
    annotation_count: int
    classes: list[str]
    splits: dict[str, int]
    checksum: str
    size_bytes: int
    imported_at: str = Field(default_factory=utc_now)
    inspection: DatasetInspection


class RegisteredWeight(BaseModel):
    """A checkpoint file inside an imported model that the user explicitly registered for a built-in adapter."""
    relative_path: str
    family: str
    weight_id: str
    registered_at: str = Field(default_factory=utc_now)


class ModelArtifactWeightRequest(BaseModel):
    relative_path: str = Field(min_length=1, max_length=1024)
    family: str = Field(min_length=1, max_length=64)


class ModelArtifact(BaseModel):
    model_artifact_id: str
    display_name: str
    provenance: Provenance
    local_root: str
    status: InspectionState
    framework: str | None = None
    architecture: str | None = None
    candidate_adapter: str | None = None
    weight_id: str | None = None
    artifact_file: str | None = None
    checksum: str
    size_bytes: int
    imported_at: str = Field(default_factory=utc_now)
    inspection: ModelInspection
    registered_weights: list[RegisteredWeight] = Field(default_factory=list)


class BenchmarkCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    dataset_id: str
    sampling_strategy: Literal["random", "stratified", "manual"] = "random"
    requested_size: int = Field(default=100, ge=1, le=100000)
    seed: int = 42
    selected_item_ids: list[str] = Field(default_factory=list)


class BenchmarkDataset(BaseModel):
    benchmark_id: str
    name: str
    dataset_id: str
    dataset_hash: str
    selected_items: list[dict[str, Any]]
    seed: int
    sampling_strategy: str
    requested_size: int
    actual_size: int
    class_distribution: dict[str, int]
    benchmark_hash: str
    created_at: str = Field(default_factory=utc_now)


class RecipeVariant(BaseModel):
    variant_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    name: str
    workflow: dict[str, Any]
    inference_configuration: dict[str, Any] = Field(default_factory=dict)
    runtime_settings: dict[str, Any] = Field(default_factory=dict)


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    benchmark_id: str
    model_artifact_id: str | None = None
    variants: list[RecipeVariant] = Field(min_length=2)


class ExperimentState(str, Enum):
    DRAFT = "DRAFT"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class InferenceExperiment(BaseModel):
    experiment_id: str
    name: str
    benchmark_id: str
    model_artifact_id: str | None = None
    variants: list[RecipeVariant]
    status: ExperimentState = ExperimentState.DRAFT
    created_at: str = Field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    cancel_requested: bool = False
    result_summary: dict[str, Any] = Field(default_factory=dict)
    failures: list[dict[str, Any]] = Field(default_factory=list)


class RecipeLifecycle(str, Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    PROMOTED = "PROMOTED"
    DEPRECATED = "DEPRECATED"


class PromoteRequest(BaseModel):
    variant_id: str
    name: str | None = None
    recipe_id: str | None = Field(default=None, pattern=r"^rcp_[A-Za-z0-9_-]+$")


class InferenceRecipe(BaseModel):
    recipe_id: str
    recipe_version: int
    name: str
    lifecycle: RecipeLifecycle
    model_artifact_id: str | None = None
    model_family: str | None = None
    weight_id: str | None = None
    weight_sha256: str | None = None
    preprocessing_graph: dict[str, Any]
    preprocessing_hash: str
    inference_parameters: dict[str, Any]
    inference_configuration_hash: str
    postprocessing_graph: dict[str, Any]
    postprocessing_hash: str
    workflow: dict[str, Any]
    workflow_id: str
    workflow_version: str
    workflow_hash: str
    benchmark_id: str
    benchmark_hash: str
    source_experiment_id: str
    runtime_capabilities: list[str]
    created_at: str = Field(default_factory=utc_now)
    promoted_at: str = Field(default_factory=utc_now)


class RecipeExecuteItem(BaseModel):
    item_id: str
    asset_id: str | None = None
    image_base64: str | None = None


class RecipeExecuteRequest(BaseModel):
    recipe_id: str
    recipe_version: int | None = None
    workflow_id: str | None = None
    workflow_version: str | None = None
    items: list[RecipeExecuteItem]
