from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sizes: list[str] = Field(min_length=1)


class ModelFamily(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str
    adapter: str
    result_type: str
    extensions: list[str]
    versions: dict[str, ModelVersion]
    requires_weights: bool = True
    prompt_modes: list[Literal["text", "point", "box"]] = Field(default_factory=list)
    # Optional Model-node fields this family's adapter actually reads; the inspector hides the rest.
    controls: list[Literal["text_threshold", "mask_threshold", "multimask"]] = Field(default_factory=list)
    # Runtime configuration shown only for families that need it. Secrets are
    # deliberately excluded: remote adapters read those from the server env.
    runtime_fields: list[Literal["ocr_language", "endpoint_url", "endpoint_model"]] = Field(default_factory=list)
    # Optional architecture-level patch choices keyed by model size. An empty
    # mapping means the family does not expose this control.
    patch_sizes: dict[str, list[int]] = Field(default_factory=dict)


class FeatureExtractorDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str
    family: str
    stages: list[str] = Field(min_length=1)
    version: str
    variant: str


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    families: dict[str, ModelFamily]
    feature_extractors: dict[str, FeatureExtractorDefinition] = Field(default_factory=dict)


class WeightRecord(BaseModel):
    weight_id: str
    family: str
    filename: str
    sha256: str
    size_bytes: int
    path: str = Field(exclude=True)
