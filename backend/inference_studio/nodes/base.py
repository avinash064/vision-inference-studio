from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DataType = str


class PortDefinition(BaseModel):
    id: str
    label: str
    data_types: list[DataType]
    required: bool = True
    chainner_type: Any = None


class ConfigField(BaseModel):
    id: str
    label: str
    kind: Literal["string", "textarea", "number", "boolean", "select", "asset", "weight", "color", "file", "directory"]
    required: bool = False
    default: Any = None
    options: list[Any] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    placeholder: str | None = None
    condition: Any = None
    chainner_metadata: dict[str, Any] = Field(default_factory=dict)


class NodeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    label: str
    category: str
    description: str
    inputs: list[PortDefinition] = Field(default_factory=list)
    outputs: list[PortDefinition] = Field(default_factory=list)
    config: list[ConfigField] = Field(default_factory=list)
    provider: str = "vision"
    package: str = "Vision"
    group: str = ""
    schema_id: str | None = None
    kind: str = "regularNode"
    available: bool = True
    availability: str = "available"
    availability_reason: str | None = None
    side_effects: bool = False
    deprecated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeInstance(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    type: str
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})
    config: dict[str, Any] = Field(default_factory=dict)
