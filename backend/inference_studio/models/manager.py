from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel

from inference_studio.adapters import AdapterRegistry, LoadedModel, ModelAdapter, ModelConfig

from .registry import ModelRegistry
from .weights import WeightStore


class ModelState(str, Enum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    LOADED = "LOADED"
    ERROR = "ERROR"


class ModelStatus(BaseModel):
    model_id: str
    state: ModelState
    family: str
    adapter: str
    device: str
    weight_id: str
    error: str | None = None


class ManagedModel:
    def __init__(self, config: ModelConfig, adapter: ModelAdapter) -> None:
        self.config = config
        self.adapter = adapter
        self.state = ModelState.UNLOADED
        self.loaded: LoadedModel | None = None
        self.error: str | None = None
        self.lock = asyncio.Lock()


class ModelManager:
    def __init__(self, registry: ModelRegistry, weights: WeightStore, adapters: AdapterRegistry) -> None:
        self.registry = registry
        self.weights = weights
        self.adapters = adapters
        self._models: dict[str, ManagedModel] = {}

    @staticmethod
    def model_id(config: ModelConfig) -> str:
        return f"{config.family}:{config.version}:{config.size}:{config.weight_id}:{config.device}:{config.precision}"

    def _entry(self, config: ModelConfig) -> ManagedModel:
        model_id = self.model_id(config)
        if model_id not in self._models:
            family = self.registry.validate_selection(config.family, config.version, config.size)
            self._models[model_id] = ManagedModel(config, self.adapters.get(family.adapter))
        return self._models[model_id]

    async def load(self, config: ModelConfig) -> LoadedModel:
        entry = self._entry(config)
        async with entry.lock:
            if entry.state == ModelState.LOADED and entry.loaded is not None:
                return entry.loaded
            entry.state, entry.error = ModelState.LOADING, None
            try:
                weight = self.weights.get(config.weight_id, config.family)
                entry.loaded = await asyncio.to_thread(entry.adapter.load, config, __import__("pathlib").Path(weight.path))
                entry.state = ModelState.LOADED
                return entry.loaded
            except Exception as exc:
                entry.state, entry.error = ModelState.ERROR, str(exc)
                raise

    async def predict(self, config: ModelConfig, image: Any):
        entry = self._entry(config)
        loaded = await self.load(config)
        return await asyncio.to_thread(entry.adapter.predict, loaded, image)

    async def unload(self, config: ModelConfig) -> None:
        entry = self._entry(config)
        async with entry.lock:
            if entry.loaded is not None:
                await asyncio.to_thread(entry.adapter.unload, entry.loaded)
            entry.loaded, entry.state, entry.error = None, ModelState.UNLOADED, None

    def status(self) -> list[ModelStatus]:
        return [ModelStatus(model_id=model_id, state=entry.state, family=entry.config.family, adapter=entry.adapter.name, device=entry.config.device, weight_id=entry.config.weight_id, error=entry.error) for model_id, entry in self._models.items()]

