from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from inference_studio.results import FeatureResult

from .base import LoadedModel, ModelAdapter, ModelConfig


class DinoV3Adapter(ModelAdapter):
    name = "dinov3"
    _models = {"vit-s": "facebook/dinov3-vits16-pretrain-lvd1689m", "vit-b": "facebook/dinov3-vitb16-pretrain-lvd1689m", "vit-l": "facebook/dinov3-vitl16-pretrain-lvd1689m"}

    def load(self, config: ModelConfig, weight_path: Path) -> LoadedModel:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("DINOv3 runtime is not installed; install requirements-runtimes.txt") from exc
        identifier = self._models[config.size]
        processor = AutoImageProcessor.from_pretrained(identifier)
        network = AutoModel.from_pretrained(identifier)
        # A supplied state-dict is applied explicitly; upload alone never loads code.
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        network.load_state_dict(state, strict=False)
        network.to(config.device).eval()
        if config.precision == "fp16":
            network.half()
        return LoadedModel(handle=(processor, network), config=config, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, input_data: np.ndarray):
        processor, network = model.handle
        rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(model.config.device) for key, value in inputs.items()}
        if model.config.precision == "fp16":
            inputs = {key: value.half() for key, value in inputs.items()}
        return inputs

    def infer(self, model: LoadedModel, tensor):
        import torch
        _, network = model.handle
        with torch.inference_mode():
            return network(**tensor).last_hidden_state

    def postprocess(self, model: LoadedModel, output) -> FeatureResult:
        array = output.detach().float().cpu().numpy()
        summary = {"min": float(array.min()), "max": float(array.max()), "mean": float(array.mean()), "std": float(array.std())}
        return FeatureResult(shape=list(array.shape), values=array.reshape(-1)[:4096].tolist(), summary=summary)

    def unload(self, model: LoadedModel) -> None:
        _, network = model.handle
        del network
        super().unload(model)
