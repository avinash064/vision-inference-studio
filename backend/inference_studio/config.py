from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    project_root: Path
    workspace_dir: Path
    weights_dir: Path
    workflows_dir: Path
    models_file: Path
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024

    @classmethod
    def load(cls) -> "Settings":
        default_root = Path(__file__).resolve().parents[2]
        root = Path(os.getenv("VIS_PROJECT_ROOT", default_root)).resolve()
        return cls(
            project_root=root,
            workspace_dir=root / "workspace",
            weights_dir=root / "weights",
            workflows_dir=root / "workflows",
            models_file=root / "config" / "models.yaml",
            max_upload_bytes=int(os.getenv("VIS_MAX_UPLOAD_BYTES", 2 * 1024**3)),
        )

    def prepare(self) -> None:
        for path in (self.workspace_dir / "assets", self.workspace_dir / "previews", self.workspace_dir / "benchmarks", self.weights_dir, self.workflows_dir):
            path.mkdir(parents=True, exist_ok=True)
