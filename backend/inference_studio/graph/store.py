from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from .schema import Workflow


class WorkflowStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, workflow: Workflow, workflow_id: str | None = None) -> str:
        value = workflow_id or f"wf_{uuid.uuid4().hex}"
        if not re.fullmatch(r"wf_[A-Za-z0-9]+", value):
            raise ValueError("Invalid workflow id")
        (self.root / f"{value}.json").write_text(workflow.model_dump_json(indent=2), encoding="utf-8")
        return value

    def get(self, workflow_id: str) -> Workflow:
        if not re.fullmatch(r"wf_[A-Za-z0-9]+", workflow_id):
            raise ValueError("Invalid workflow id")
        path = self.root / f"{workflow_id}.json"
        if not path.is_file():
            raise ValueError(f"Unknown workflow: {workflow_id}")
        return Workflow.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        values = []
        for path in sorted(self.root.glob("wf_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                values.append({"workflow_id": path.stem, "name": data.get("name", "Untitled workflow"), "version": data.get("version", "0.2")})
            except (OSError, json.JSONDecodeError):
                continue
        return values
