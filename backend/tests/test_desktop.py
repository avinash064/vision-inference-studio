from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from inference_studio.config import Settings
from inference_studio.main import create_app
from inference_studio.runtime_check import REQUIRED_IMPORTS


def test_desktop_paths_use_xdg_data(monkeypatch, tmp_path: Path):
    data = tmp_path / "data"
    config = tmp_path / "config"
    monkeypatch.setenv("VIS_DATA_DIR", str(data))
    monkeypatch.setenv("VIS_CONFIG_DIR", str(config))
    settings = Settings.load()
    assert settings.weights_dir == data / "weights"
    assert settings.workflows_dir == data / "workflows"
    assert settings.workspace_dir == data / "workspace"
    assert settings.benchmarks_dir == data / "benchmarks"
    assert settings.models_file == config / "models.yaml"


def test_desktop_http_requires_session_token(monkeypatch, tmp_path: Path):
    root = Path(__file__).parents[2]
    monkeypatch.setenv("VIS_DESKTOP_MODE", "1")
    monkeypatch.setenv("VIS_API_TOKEN", "test-session-token")
    settings = Settings(
        project_root=root,
        workspace_dir=tmp_path / "workspace",
        weights_dir=tmp_path / "weights",
        workflows_dir=tmp_path / "workflows",
        benchmarks_dir=tmp_path / "benchmarks",
        models_file=root / "config" / "models.yaml",
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 401
        response = client.get("/health", headers={"X-Vision-Session": "test-session-token"})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_runtime_check_covers_all_required_engines():
    assert {"fastapi", "uvicorn", "pydantic", "cv2", "numpy", "torch", "torchvision", "ultralytics", "rfdetr", "onnxruntime", "transformers", "safetensors", "groundingdino", "clip", "chainner_ext"} == set(REQUIRED_IMPORTS)
