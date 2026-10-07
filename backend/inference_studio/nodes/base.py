from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DataType = Literal["image", "video", "frame", "detections", "bboxes", "mask", "segmentation", "classification", "feature_map", "embedding", "tensor", "heatmap", "depth", "weights", "number", "string", "boolean"]


class PortDefinition(BaseModel):
    id: str
    label: str
    data_types: list[DataType]
    required: bool = True


class ConfigField(BaseModel):
    id: str
    label: str
    kind: Literal["string", "number", "boolean", "select", "asset", "weight"]
    required: bool = False
    default: Any = None
    options: list[str] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None


class NodeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    label: str
    category: Literal["sources", "models", "processing", "outputs"]
    description: str
    inputs: list[PortDefinition] = Field(default_factory=list)
    outputs: list[PortDefinition] = Field(default_factory=list)
    config: list[ConfigField] = Field(default_factory=list)


class NodeInstance(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    type: str
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})
    config: dict[str, Any] = Field(default_factory=dict)

