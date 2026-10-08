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
    prompt_modes: list[Literal["text", "point", "box"]] = Field(default_factory=list)
    # Optional Model-node fields this family's adapter actually reads; the inspector hides the rest.
    controls: list[Literal["text_threshold", "mask_threshold", "multimask"]] = Field(default_factory=list)


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    families: dict[str, ModelFamily]


class WeightRecord(BaseModel):
    weight_id: str
    family: str
    filename: str
    sha256: str
    size_bytes: int
    path: str = Field(exclude=True)
