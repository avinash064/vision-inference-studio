from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def load_env_file(path: Path) -> list[str]:
    """Load KEY=VALUE lines (e.g. HF_TOKEN for gated Hugging Face models) without overriding the real environment."""

    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


class Settings(BaseModel):
    project_root: Path
    workspace_dir: Path
    weights_dir: Path
    workflows_dir: Path
    benchmarks_dir: Path
    models_file: Path
    max_upload_bytes: int = 5 * 1024 * 1024 * 1024
    chainner_src_dir: Path | None = None
    reports_dir: Path | None = None
    config_dir: Path | None = None
    feature_store_max_entries: int = 64
    feature_store_max_bytes: int = 2 * 1024**3
    geometries_dir_override: Path | None = None
    public_hub_root: Path | None = None
    public_dataset_cache: Path | None = None
    public_model_cache: Path | None = None
    public_hub_max_cache_bytes: int = 50 * 1024**3
    public_hub_max_concurrent_downloads: int = 2
    public_hub_max_extract_bytes: int = 100 * 1024**3
    public_hub_max_file_bytes: int = 20 * 1024**3
    recipe_input_roots: list[Path] = Field(default_factory=list)
    auto_annotation_root: Path | None = None
    auto_annotation_output_root: Path | None = None
    auto_annotation_input_roots: list[Path] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        self.public_hub_root = (self.public_hub_root or self.workspace_dir / "public-hub").resolve()
        self.public_dataset_cache = (self.public_dataset_cache or self.public_hub_root / "datasets").resolve()
        self.public_model_cache = (self.public_model_cache or self.public_hub_root / "models").resolve()
        self.auto_annotation_root = (self.auto_annotation_root or self.workspace_dir / "auto-annotation").resolve()
        self.auto_annotation_output_root = (self.auto_annotation_output_root or self.auto_annotation_root / "packages").resolve()
        if not self.auto_annotation_input_roots:
            self.auto_annotation_input_roots = list(self.recipe_input_roots) or [self.project_root.resolve()]

    @property
    def geometries_dir(self) -> Path:
        """Frozen Mahalanobis geometry store (VIS_GEOMETRIES_DIR, default <data>/feature_geometries)."""
        return self.geometries_dir_override or self.weights_dir.parent / "feature_geometries"

    @classmethod
    def load(cls) -> "Settings":
        default_root = Path(__file__).resolve().parents[2]
        load_env_file(Path(os.getenv("VIS_ENV_FILE", default_root / ".env")).expanduser())
        root = Path(os.getenv("VIS_PROJECT_ROOT", default_root)).resolve()
        data_dir = Path(os.getenv("VIS_DATA_DIR", root)).expanduser().resolve()
        config_dir = Path(os.getenv("VIS_CONFIG_DIR", root / "config")).expanduser().resolve()
        workspace_dir = Path(os.getenv("VIS_WORKSPACE_DIR", data_dir / "workspace")).expanduser().resolve()
        # Vendored snapshot first (third_party/chainner, GPL-3.0); the monorepo sibling src/ is a fallback.
        default_chainner = next((path for path in (root / "third_party" / "chainner" / "src", root.parent / "src") if path.is_dir()), root.parent / "src")
        chainner_value = os.getenv("VIS_CHAINNER_SRC_DIR")
        hub_root = Path(os.getenv("PUBLIC_HUB_CACHE", workspace_dir / "public-hub")).expanduser().resolve()
        recipe_roots = [Path(value).expanduser().resolve() for value in os.getenv("VIS_RECIPE_INPUT_ROOTS", "").split(os.pathsep) if value]
        annotation_roots = [Path(value).expanduser().resolve() for value in os.getenv("VIS_AUTO_ANNOTATION_INPUT_ROOTS", "").split(os.pathsep) if value]
        return cls(
            project_root=root,
            workspace_dir=workspace_dir,
            weights_dir=Path(os.getenv("VIS_WEIGHTS_DIR", data_dir / "weights")).expanduser().resolve(),
            workflows_dir=Path(os.getenv("VIS_WORKFLOWS_DIR", data_dir / "workflows")).expanduser().resolve(),
            benchmarks_dir=Path(os.getenv("VIS_BENCHMARKS_DIR", data_dir / "benchmarks")).expanduser().resolve(),
            models_file=Path(os.getenv("VIS_MODELS_FILE", config_dir / "models.yaml")).expanduser().resolve(),
            max_upload_bytes=int(os.getenv("VIS_MAX_UPLOAD_BYTES", 5 * 1024**3)),
            chainner_src_dir=Path(chainner_value).expanduser().resolve() if chainner_value else (default_chainner.resolve() if default_chainner.is_dir() else None),
            reports_dir=Path(os.getenv("VIS_REPORTS_DIR", workspace_dir / "reports")).expanduser().resolve(),
            config_dir=config_dir,
            feature_store_max_entries=int(os.getenv("VIS_FEATURE_STORE_MAX_ENTRIES", "64")),
            feature_store_max_bytes=int(os.getenv("VIS_FEATURE_STORE_MAX_BYTES", str(2 * 1024**3))),
            geometries_dir_override=Path(os.getenv("VIS_GEOMETRIES_DIR", data_dir / "feature_geometries")).expanduser().resolve(),
            public_hub_root=hub_root,
            public_dataset_cache=Path(os.getenv("PUBLIC_DATASET_CACHE", hub_root / "datasets")).expanduser().resolve(),
            public_model_cache=Path(os.getenv("PUBLIC_MODEL_CACHE", hub_root / "models")).expanduser().resolve(),
            public_hub_max_cache_bytes=int(float(os.getenv("PUBLIC_HUB_MAX_CACHE_GB", "50")) * 1024**3),
            public_hub_max_concurrent_downloads=max(1, int(os.getenv("PUBLIC_HUB_MAX_CONCURRENT_DOWNLOADS", "2"))),
            public_hub_max_extract_bytes=int(float(os.getenv("PUBLIC_HUB_MAX_EXTRACT_GB", "100")) * 1024**3),
            public_hub_max_file_bytes=int(float(os.getenv("PUBLIC_HUB_MAX_FILE_GB", "20")) * 1024**3),
            recipe_input_roots=recipe_roots,
            auto_annotation_root=Path(os.getenv("VIS_AUTO_ANNOTATION_ROOT", workspace_dir / "auto-annotation")).expanduser().resolve(),
            auto_annotation_output_root=Path(os.getenv("VIS_AUTO_ANNOTATION_OUTPUT_ROOT", workspace_dir / "auto-annotation" / "packages")).expanduser().resolve(),
            auto_annotation_input_roots=annotation_roots or recipe_roots or [root],
        )

    def prepare(self) -> None:
        for path in (self.workspace_dir / "assets", self.workspace_dir / "previews", self.weights_dir, self.workflows_dir, self.benchmarks_dir, self.reports_dir or self.workspace_dir / "reports", self.geometries_dir, self.public_hub_root, self.public_dataset_cache, self.public_model_cache, self.auto_annotation_root, self.auto_annotation_output_root):
            path.mkdir(parents=True, exist_ok=True)
