from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from inference_studio.adapters import AdapterRegistry, InferenceRequest, LoadedModel, ModelAdapter, ModelConfig, ModelKey
from inference_studio.features.extractors import (
    FeatureExtractor, FeatureExtractorRegistry, FeatureModelKey, LoadedFeatureModel,
)

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
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManagedModel:
    def __init__(self, key: ModelKey, adapter: ModelAdapter) -> None:
        self.key = key
        self.adapter = adapter
        self.state = ModelState.UNLOADED
        self.loaded: LoadedModel | None = None
        self.error: str | None = None
        self.load_lock = asyncio.Lock()
        self.max_concurrency = max(1, adapter.max_concurrency)
        self.inference_gate = asyncio.Semaphore(self.max_concurrency)


class ManagedFeatureModel:
    def __init__(self, key: FeatureModelKey, extractor: FeatureExtractor) -> None:
        self.key = key
        self.extractor = extractor
        self.state = ModelState.UNLOADED
        self.loaded: LoadedFeatureModel | None = None
        self.error: str | None = None
        self.load_count = 0
        self.load_lock = asyncio.Lock()
        # RF-DETR wrappers expose mutable hook/forward internals.  One forward
        # at a time also makes shared srcs/masks/poss ownership deterministic.
        self.inference_gate = asyncio.Semaphore(1)


class ModelManager:
    def __init__(self, registry: ModelRegistry, weights: WeightStore, adapters: AdapterRegistry, feature_extractors: FeatureExtractorRegistry | None = None) -> None:
        self.registry = registry
        self.weights = weights
        self.adapters = adapters
        self.feature_extractors = feature_extractors or FeatureExtractorRegistry()
        self._models: dict[str, ManagedModel] = {}
        self._feature_models: dict[str, ManagedFeatureModel] = {}

    @staticmethod
    def model_id(value: ModelConfig | ModelKey) -> str:
        key = value.model_key() if isinstance(value, ModelConfig) else value
        return f"{key.family}:{key.version}:{key.size}:patch{key.patch_size}:{key.weight_id}:{key.device}:{key.precision}:{key.ocr_language}:{key.endpoint_url}:{key.endpoint_model}"

    def _entry(self, value: ModelConfig | ModelKey) -> ManagedModel:
        key = value.model_key() if isinstance(value, ModelConfig) else value
        model_id = self.model_id(key)
        if model_id not in self._models:
            family = self.registry.validate_selection(key.family, key.version, key.size)
            self._models[model_id] = ManagedModel(key, self.adapters.get(family.adapter))
        return self._models[model_id]

    async def load(self, value: ModelConfig | ModelKey) -> LoadedModel:
        key = value.model_key() if isinstance(value, ModelConfig) else value
        entry = self._entry(key)
        async with entry.load_lock:
            if entry.state == ModelState.LOADED and entry.loaded is not None:
                return entry.loaded
            entry.state, entry.error = ModelState.LOADING, None
            try:
                family = self.registry.validate_selection(key.family, key.version, key.size)
                if family.requires_weights:
                    weight = self.weights.get(key.weight_id, key.family)
                    entry.loaded = await asyncio.to_thread(entry.adapter.load, key, __import__("pathlib").Path(weight.path))
                    entry.loaded.checkpoint_sha256 = weight.sha256
                else:
                    entry.loaded = await asyncio.to_thread(entry.adapter.load, key, __import__("pathlib").Path())
                entry.state = ModelState.LOADED
                return entry.loaded
            except Exception as exc:
                entry.state, entry.error = ModelState.ERROR, str(exc)
                raise

    async def predict(self, config: ModelConfig | ModelKey, image: Any, request: InferenceRequest | None = None):
        key = config.model_key() if isinstance(config, ModelConfig) else config
        effective_request = config.inference_request() if isinstance(config, ModelConfig) else (request or InferenceRequest())
        entry = self._entry(key)
        async with entry.inference_gate:
            loaded = await self.load(key)
            return await asyncio.to_thread(entry.adapter.predict, loaded, effective_request, image)

    @staticmethod
    def feature_model_id(key: FeatureModelKey) -> str:
        return (
            f"feature:{key.family}:{key.version}:{key.variant}:"
            f"{key.weight_id}:{key.device}:{key.precision}"
        )

    def feature_key(self, provider: str, weight_id: str, device: str = "cpu", precision: str = "fp32") -> FeatureModelKey:
        return self.feature_extractors.get(provider).model_key(weight_id, device, precision)

    def _feature_entry(self, key: FeatureModelKey, provider: str) -> ManagedFeatureModel:
        model_id = self.feature_model_id(key)
        if model_id not in self._feature_models:
            self._feature_models[model_id] = ManagedFeatureModel(key, self.feature_extractors.get(provider))
        entry = self._feature_models[model_id]
        if entry.extractor.name != provider:
            raise ValueError(f"Feature model key collision between {entry.extractor.name} and {provider}")
        return entry

    async def load_feature(
        self, provider: str, weight_id: str, device: str = "cpu", precision: str = "fp32"
    ) -> LoadedFeatureModel:
        key = self.feature_key(provider, weight_id, device, precision)
        entry = self._feature_entry(key, provider)
        async with entry.load_lock:
            if entry.state == ModelState.LOADED and entry.loaded is not None:
                return entry.loaded
            entry.state, entry.error = ModelState.LOADING, None
            try:
                weight = self.weights.get(weight_id, key.family)
                entry.loaded = await asyncio.to_thread(
                    entry.extractor.load, key, __import__("pathlib").Path(weight.path), weight.sha256
                )
                entry.load_count += 1
                entry.state = ModelState.LOADED
                return entry.loaded
            except Exception as exc:
                entry.state, entry.error = ModelState.ERROR, str(exc)
                raise

    async def extract_feature(
        self,
        provider: str,
        weight_id: str,
        image: Any,
        stage: str,
        device: str = "cpu",
        precision: str = "fp32",
    ):
        key = self.feature_key(provider, weight_id, device, precision)
        entry = self._feature_entry(key, provider)
        async with entry.inference_gate:
            loaded = await self.load_feature(provider, weight_id, device, precision)
            if stage not in entry.extractor.stages(loaded):
                raise ValueError(
                    f"{provider} stage {stage!r} is unsupported; choose one of "
                    f"{', '.join(entry.extractor.stages(loaded))}"
                )
            return await asyncio.to_thread(entry.extractor.extract, loaded, image, stage)

    def feature_load_count(
        self, provider: str, weight_id: str, device: str = "cpu", precision: str = "fp32"
    ) -> int:
        key = self.feature_key(provider, weight_id, device, precision)
        entry = self._feature_models.get(self.feature_model_id(key))
        return entry.load_count if entry is not None else 0

    async def unload(self, config: ModelConfig | ModelKey) -> None:
        entry = self._entry(config)
        await self._drain_and_unload(entry)

    async def unload_all(self) -> None:
        for entry in list(self._models.values()):
            await self._drain_and_unload(entry)
        for entry in list(self._feature_models.values()):
            await self._drain_and_unload_feature(entry)

    @staticmethod
    async def _drain_and_unload(entry: ManagedModel) -> None:
        for _ in range(entry.max_concurrency):
            await entry.inference_gate.acquire()
        try:
            async with entry.load_lock:
                if entry.loaded is not None:
                    await asyncio.to_thread(entry.adapter.unload, entry.loaded)
                entry.loaded, entry.state, entry.error = None, ModelState.UNLOADED, None
        finally:
            for _ in range(entry.max_concurrency):
                entry.inference_gate.release()

    @staticmethod
    async def _drain_and_unload_feature(entry: ManagedFeatureModel) -> None:
        await entry.inference_gate.acquire()
        try:
            async with entry.load_lock:
                if entry.loaded is not None:
                    await asyncio.to_thread(entry.extractor.unload, entry.loaded)
                entry.loaded, entry.state, entry.error = None, ModelState.UNLOADED, None
        finally:
            entry.inference_gate.release()

    def status(self) -> list[ModelStatus]:
        ordinary = [ModelStatus(model_id=model_id, state=entry.state, family=entry.key.family, adapter=entry.adapter.name, device=entry.key.device, weight_id=entry.key.weight_id, error=entry.error, metadata=entry.adapter.metadata(entry.loaded) if entry.loaded is not None else {}) for model_id, entry in self._models.items()]
        features = [
            ModelStatus(
                model_id=model_id, state=entry.state, family=entry.key.family,
                adapter=entry.extractor.name, device=entry.key.device,
                weight_id=entry.key.weight_id, error=entry.error,
                metadata={
                    "feature_provider": entry.extractor.name,
                    "version": entry.key.version,
                    "variant": entry.key.variant,
                    "precision": entry.key.precision,
                    "load_count": entry.load_count,
                    "checkpoint_sha256": entry.loaded.checkpoint_sha256 if entry.loaded else None,
                },
            )
            for model_id, entry in self._feature_models.items()
        ]
        return [*ordinary, *features]

    def state_for(self, config: ModelConfig) -> ModelState:
        entry = self._models.get(self.model_id(config))
        return entry.state if entry is not None else ModelState.UNLOADED
