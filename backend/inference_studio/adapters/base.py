from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from inference_studio.results import InferenceMetrics, InferenceResult


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: str
    version: str
    size: str
    weight_id: str
    device: str = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"
    confidence: float = Field(default=0.25, ge=0, le=1)


@dataclass
class LoadedModel:
    handle: Any
    config: ModelConfig
    weight_path: Path


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def load(self, config: ModelConfig, weight_path: Path) -> LoadedModel: ...

    @abstractmethod
    def preprocess(self, model: LoadedModel, input_data: np.ndarray) -> Any: ...

    @abstractmethod
    def infer(self, model: LoadedModel, tensor: Any) -> Any: ...

    @abstractmethod
    def postprocess(self, model: LoadedModel, output: Any) -> InferenceResult: ...

    def predict(self, model: LoadedModel, input_data: np.ndarray) -> InferenceResult:
        started = self.timer()
        phase = perf_counter()
        tensor = self.preprocess(model, input_data)
        preprocess_ms = (perf_counter() - phase) * 1000
        phase = perf_counter()
        raw = self.infer(model, tensor)
        inference_ms = (perf_counter() - phase) * 1000
        phase = perf_counter()
        result = self.postprocess(model, raw)
        postprocess_ms = (perf_counter() - phase) * 1000
        result.metrics = self.metrics(started, inference_ms, preprocess_ms, postprocess_ms)
        return result

    def unload(self, model: LoadedModel) -> None:
        del model.handle
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def metadata(self, model: LoadedModel) -> dict[str, Any]:
        return {"adapter": self.name, "family": model.config.family, "version": model.config.version, "size": model.config.size, "device": model.config.device, "precision": model.config.precision, "weight": model.weight_path.name}

    @staticmethod
    def timer() -> float:
        return perf_counter()

    @staticmethod
    def metrics(start: float, inference_ms: float, preprocess_ms: float = 0, postprocess_ms: float = 0) -> InferenceMetrics:
        gpu_memory = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
        except ImportError:
            pass
        return InferenceMetrics(preprocess_ms=preprocess_ms, inference_ms=inference_ms, postprocess_ms=postprocess_ms, total_ms=(perf_counter() - start) * 1000, gpu_memory_mb=gpu_memory)
