from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from inference_studio.runtime.artifacts import FeatureArtifact, FeatureMetadata, FeatureSpaceSignature
from inference_studio.runtime.features import summary

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey


class DinoV3Adapter(ModelAdapter):
    name = "dinov3"
    _models = {"vit-s": "facebook/dinov3-vits16-pretrain-lvd1689m", "vit-b": "facebook/dinov3-vitb16-pretrain-lvd1689m", "vit-l": "facebook/dinov3-vitl16-pretrain-lvd1689m"}

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("DINOv3 runtime is not installed; install requirements-runtimes.txt") from exc
        identifier = self._models[key.size]
        processor = AutoImageProcessor.from_pretrained(identifier)
        network = AutoModel.from_pretrained(identifier)
        # A supplied state-dict is applied explicitly; upload alone never loads code.
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        network.load_state_dict(state, strict=False)
        network.to(key.device).eval()
        if key.precision == "fp16":
            network.half()
        return LoadedModel(handle=(processor, network, identifier), key=key, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray):
        processor, network, _ = model.handle
        rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(model.key.device) for key, value in inputs.items()}
        if model.key.precision == "fp16":
            inputs = {key: value.half() for key, value in inputs.items()}
        return inputs, input_data.shape[:2]

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor):
        import torch
        _, network, _ = model.handle
        inputs, source_size = tensor
        with torch.inference_mode():
            output = network(**inputs).last_hidden_state
        return output, tuple(inputs["pixel_values"].shape[-2:]), source_size

    def postprocess(self, model: LoadedModel, request: InferenceRequest, output) -> FeatureArtifact:
        tokens, (input_height, input_width), (source_height, source_width) = output
        _, network, identifier = model.handle
        patch = getattr(network.config, "patch_size", 16)
        patch_h, patch_w = (patch, patch) if isinstance(patch, int) else tuple(patch)
        grid_height, grid_width = input_height // patch_h, input_width // patch_w
        patch_count = grid_height * grid_width
        if tokens.shape[1] < patch_count:
            raise RuntimeError(f"DINO returned {tokens.shape[1]} tokens for a {grid_height}x{grid_width} patch grid")
        register_tokens = getattr(network.config, "num_register_tokens", None)
        if register_tokens is not None:
            expected_tokens = patch_count + 1 + int(register_tokens)
            if tokens.shape[1] != expected_tokens:
                raise RuntimeError(f"DINO token layout mismatch: expected {expected_tokens} tokens ({patch_count} patches + CLS/register tokens), got {tokens.shape[1]}")
        # DINO special/register tokens precede spatial patch tokens.
        spatial = tokens[0, -patch_count:, :].reshape(grid_height, grid_width, -1).contiguous()
        dtype = str(spatial.dtype).removeprefix("torch.")
        signature = FeatureSpaceSignature(
            extractor_family=model.key.family,
            extractor_version=model.key.version,
            extractor_variant=model.key.size,
            checkpoint_sha256=model.checkpoint_sha256 or model.key.weight_id,
            feature_stage="last_hidden_state",
            spatial_height=grid_height,
            spatial_width=grid_width,
            channels=spatial.shape[-1],
            input_height=input_height,
            input_width=input_width,
            stride=float(patch_h),
            patch_size=patch_h,
            preprocessing_signature=identifier,
            normalization="processor-default",
        )
        metadata = FeatureMetadata(
            shape=list(spatial.shape),
            dtype=dtype,
            device=str(spatial.device),
            memory_bytes=spatial.numel() * spatial.element_size(),
            source_width=source_width,
            source_height=source_height,
            feature_width=grid_width,
            feature_height=grid_height,
            channels=spatial.shape[-1],
            extractor_family=model.key.family,
            extractor_version=model.key.version,
            extractor_variant=model.key.size,
            feature_stage="last_hidden_state",
            stride=float(patch_h),
            patch_size=patch_h,
            signature=signature,
        )
        return FeatureArtifact(spatial, metadata, summary(spatial))

    def unload(self, model: LoadedModel) -> None:
        _, network, _ = model.handle
        del network
        super().unload(model)
