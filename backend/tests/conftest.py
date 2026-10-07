from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest

from inference_studio.assets import AssetStore
from inference_studio.models import ModelRegistry, WeightStore


@pytest.fixture
def model_registry() -> ModelRegistry:
    return ModelRegistry.load(Path(__file__).parents[2] / "config" / "models.yaml")


@pytest.fixture
def stores(tmp_path: Path, model_registry: ModelRegistry):
    return WeightStore(tmp_path / "weights", model_registry, 10_000_000), AssetStore(tmp_path / "assets", 10_000_000)


@pytest.fixture
def image_bytes() -> bytes:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[10:35, 15:42] = (20, 180, 230)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def registered_files(stores, image_bytes):
    weights, assets = stores
    weight = weights.save(BytesIO(b"test weight bytes"), "model.pt", "yolo")
    asset = assets.save(BytesIO(image_bytes), "sample.png", "image")
    return weight, asset

