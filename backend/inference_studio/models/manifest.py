from __future__ import annotations

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

