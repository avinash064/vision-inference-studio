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
    text_prompt: str = ""
    text_threshold: float = Field(default=0.25, ge=0, le=1)
    prompt_mode: Literal["", "text", "point", "box"] = ""
    point_prompt: str = ""
    point_foreground: bool = True
    box_prompt: str = ""
    mask_threshold: float = Field(default=0.0, ge=-10, le=10)
    multimask: bool = False


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
        cuda_device = None
        try:
            import torch
            if model.config.device.startswith("cuda") and torch.cuda.is_available():
                cuda_device = torch.device(model.config.device)
                torch.cuda.synchronize(cuda_device)
                torch.cuda.reset_peak_memory_stats(cuda_device)
        except (ImportError, RuntimeError):
            cuda_device = None
        started = self.timer()
        phase = perf_counter()
        tensor = self.preprocess(model, input_data)
        preprocess_ms = (perf_counter() - phase) * 1000
        phase = perf_counter()
        raw = self.infer(model, tensor)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        inference_ms = (perf_counter() - phase) * 1000
        phase = perf_counter()
        result = self.postprocess(model, raw)
        postprocess_ms = (perf_counter() - phase) * 1000
        result.metrics = self.metrics(started, inference_ms, preprocess_ms, postprocess_ms, cuda_device)
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
        metadata: dict[str, Any] = {"adapter": self.name, "family": model.config.family, "version": model.config.version, "size": model.config.size, "device": model.config.device, "precision": model.config.precision, "weight": model.weight_path.name}
        try:
            import torch
            if model.config.device.startswith("cuda") and torch.cuda.is_available():
                device = torch.device(model.config.device)
                metadata.update(device_name=torch.cuda.get_device_name(device), gpu_memory_mb=torch.cuda.memory_allocated(device) / 1024**2, gpu_reserved_mb=torch.cuda.memory_reserved(device) / 1024**2, gpu_peak_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2)
        except (ImportError, RuntimeError):
            pass
        return metadata

    @staticmethod
    def timer() -> float:
        return perf_counter()

    @staticmethod
    def metrics(start: float, inference_ms: float, preprocess_ms: float = 0, postprocess_ms: float = 0, cuda_device: Any = None) -> InferenceMetrics:
        allocated = reserved = peak = 0.0
        device_name = None
        try:
            import torch
            if cuda_device is not None and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(cuda_device) / 1024**2
                reserved = torch.cuda.memory_reserved(cuda_device) / 1024**2
                peak = torch.cuda.max_memory_allocated(cuda_device) / 1024**2
                device_name = torch.cuda.get_device_name(cuda_device)
        except ImportError:
            pass
        return InferenceMetrics(preprocess_ms=preprocess_ms, inference_ms=inference_ms, postprocess_ms=postprocess_ms, total_ms=(perf_counter() - start) * 1000, gpu_memory_mb=allocated, gpu_reserved_mb=reserved, gpu_peak_memory_mb=peak, device_name=device_name)
