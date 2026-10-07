from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from inference_studio.nodes import NodeInstance


class Edge(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    source: str
    source_handle: str
    target: str
    target_handle: str


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "0.1"
    name: str = "Untitled workflow"
    nodes: list[NodeInstance] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    code: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None


class ValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue]
    order: list[str] = Field(default_factory=list)

