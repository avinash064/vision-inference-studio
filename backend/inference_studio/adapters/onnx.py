from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from inference_studio.results import TensorResult

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey


class OnnxAdapter(ModelAdapter):
    name = "onnx"

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is not installed; install requirements-runtimes.txt") from exc
        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if key.device.startswith("cuda") and "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        return LoadedModel(handle=ort.InferenceSession(str(weight_path), providers=providers), key=key, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray) -> np.ndarray:
        input_meta = model.handle.get_inputs()[0]
        shape = input_meta.shape
        target_h = shape[-2] if isinstance(shape[-2], int) else input_data.shape[0]
        target_w = shape[-1] if isinstance(shape[-1], int) else input_data.shape[1]
        rgb = cv2.cvtColor(cv2.resize(input_data, (target_w, target_h)), cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        if model.key.precision == "fp16":
            tensor = tensor.astype(np.float16)
        return tensor

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor: np.ndarray):
        return model.handle.run(None, {model.handle.get_inputs()[0].name: tensor})[0]

    def postprocess(self, model: LoadedModel, request: InferenceRequest, output) -> TensorResult:
        flat = np.asarray(output).reshape(-1)
        return TensorResult(shape=list(output.shape), dtype=str(output.dtype), values=flat[:4096].astype(float).tolist())
