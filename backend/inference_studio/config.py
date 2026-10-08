from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    project_root: Path
    workspace_dir: Path
    weights_dir: Path
    workflows_dir: Path
    benchmarks_dir: Path
    models_file: Path
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024

    @classmethod
    def load(cls) -> "Settings":
        default_root = Path(__file__).resolve().parents[2]
        root = Path(os.getenv("VIS_PROJECT_ROOT", default_root)).resolve()
        data_dir = Path(os.getenv("VIS_DATA_DIR", root)).expanduser().resolve()
        config_dir = Path(os.getenv("VIS_CONFIG_DIR", root / "config")).expanduser().resolve()
        workspace_dir = Path(os.getenv("VIS_WORKSPACE_DIR", data_dir / "workspace")).expanduser().resolve()
        return cls(
            project_root=root,
            workspace_dir=workspace_dir,
            weights_dir=Path(os.getenv("VIS_WEIGHTS_DIR", data_dir / "weights")).expanduser().resolve(),
            workflows_dir=Path(os.getenv("VIS_WORKFLOWS_DIR", data_dir / "workflows")).expanduser().resolve(),
            benchmarks_dir=Path(os.getenv("VIS_BENCHMARKS_DIR", data_dir / "benchmarks")).expanduser().resolve(),
            models_file=Path(os.getenv("VIS_MODELS_FILE", config_dir / "models.yaml")).expanduser().resolve(),
            max_upload_bytes=int(os.getenv("VIS_MAX_UPLOAD_BYTES", 2 * 1024**3)),
        )

    def prepare(self) -> None:
        for path in (self.workspace_dir / "assets", self.workspace_dir / "previews", self.weights_dir, self.workflows_dir, self.benchmarks_dir):
            path.mkdir(parents=True, exist_ok=True)
