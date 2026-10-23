"""Official RT-DETRv2 PyTorch adapter for the BMD-45 R101VD checkpoint.

The BMD-45 ``best.pth`` file is a native checkpoint from
https://github.com/lyuwenyu/RT-DETR.  It is not an Ultralytics ``.pt`` model,
so it must be reconstructed with the matching official architecture before
its state dictionary is loaded.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference_studio.results import Detection, DetectionResult

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey


# BMD uses its COCO category ids directly (1..13).  Its checkpoint therefore
# has 14 classifier outputs, with index 0 unused by the labelled dataset.
_BMD45_CLASSES = {
    0: "background",
    1: "Hatchback",
    2: "Sedan",
    3: "SUV",
    4: "MUV",
    5: "Bus",
    6: "Truck",
    7: "Three-wheeler",
    8: "Two-wheeler",
    9: "LCV",
    10: "Mini-bus",
    11: "Tempo-traveller",
    12: "Bicycle",
    13: "Van",
}


@dataclass
class RtDetrV2Handle:
    network: Any
    postprocessor: Any
    class_names: dict[int, str]


class RtDetrV2Adapter(ModelAdapter):
    """Run official RT-DETRv2-R101VD weights with the standard detection contract."""

    name = "rtdetrv2"

    @staticmethod
    def _runtime_root() -> Path:
        source_root = Path(__file__).resolve().parents[3]
        project_root = Path(os.getenv("VIS_PROJECT_ROOT", source_root)).resolve()
        return project_root / "third_party" / "rtdetrv2_official"

    @staticmethod
    def _state_dict(checkpoint: Any) -> dict[str, Any]:
        if not isinstance(checkpoint, dict):
            raise RuntimeError("RT-DETRv2 checkpoint must contain a PyTorch state dictionary")
        ema = checkpoint.get("ema")
        state = ema.get("module") if isinstance(ema, dict) else None
        state = state or checkpoint.get("model") or checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("RT-DETRv2 checkpoint has no model/EMA state dictionary")
        if state and all(str(name).startswith("module.") for name in state):
            state = {str(name).removeprefix("module."): value for name, value in state.items()}
        return state

    @staticmethod
    def _num_classes(state: dict[str, Any]) -> int:
        classifier = state.get("decoder.enc_score_head.weight")
        if classifier is None or not hasattr(classifier, "shape") or len(classifier.shape) != 2:
            raise RuntimeError("Checkpoint is not an official RT-DETRv2 detection state dictionary")
        return int(classifier.shape[0])

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("RT-DETRv2 requires PyTorch; install requirements-runtimes.txt") from exc

        runtime_root = self._runtime_root()
        config_path = runtime_root / "configs" / "rtdetrv2" / "rtdetrv2_r101vd_6x_coco.yml"
        if not config_path.is_file():
            raise RuntimeError(f"Packaged RT-DETRv2 runtime is missing: {config_path}")
        # The vendored official package has a unique name, avoiding the generic
        # top-level `src` module used by the upstream command-line examples.
        if str(runtime_root.parent) not in sys.path:
            sys.path.insert(0, str(runtime_root.parent))
        try:
            from rtdetrv2_official.core import YAMLConfig
        except ImportError as exc:
            raise RuntimeError("The packaged official RT-DETRv2 runtime could not be imported") from exc

        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        state = self._state_dict(checkpoint)
        num_classes = self._num_classes(state)
        # Reconstruct the BMD-45 R101VD-X architecture.  Pretrained=False is
        # essential: all backbone parameters come from this checkpoint and no
        # implicit network download is allowed while a local model is loading.
        cfg = YAMLConfig(
            str(config_path),
            num_classes=num_classes,
            remap_mscoco_category=False,
            PResNet={"pretrained": False},
        )
        try:
            cfg.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint does not match the official RT-DETRv2 R101VD-X architecture used by BMD-45"
            ) from exc
        network = cfg.model.deploy().to(key.device).eval()
        postprocessor = cfg.postprocessor.deploy().to(key.device).eval()
        names = _BMD45_CLASSES if num_classes == 14 else {index: f"class {index}" for index in range(num_classes)}
        return LoadedModel(
            handle=RtDetrV2Handle(network=network, postprocessor=postprocessor, class_names=names),
            key=key,
            weight_path=weight_path,
        )

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional

        if input_data.ndim != 3 or input_data.shape[2] != 3:
            raise ValueError("RT-DETRv2 expects a BGR image with three channels")
        height, width = input_data.shape[:2]
        rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        image = functional.interpolate(image, size=(640, 640), mode="bilinear", align_corners=False)
        return {
            "image": image.to(model.key.device, non_blocking=model.key.device.startswith("cuda")),
            # The official postprocessor expects [width, height].
            "original_size": torch.tensor([[width, height]], dtype=torch.int64, device=model.key.device),
        }

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor: dict[str, Any]):
        import torch

        half = model.key.precision == "fp16" and model.key.device.startswith("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=half):
            output = model.handle.network(tensor["image"])
            labels, boxes, scores = model.handle.postprocessor(output, tensor["original_size"])
            width, height = tensor["original_size"][0]
            boxes[..., 0::2].clamp_(min=0, max=width)
            boxes[..., 1::2].clamp_(min=0, max=height)
            return labels, boxes, scores

    def postprocess(self, model: LoadedModel, request: InferenceRequest, output: Any) -> DetectionResult:
        labels, boxes, scores = output
        detections: list[Detection] = []
        for class_id, box, confidence in zip(
            labels[0].detach().cpu().tolist(),
            boxes[0].detach().cpu().tolist(),
            scores[0].detach().cpu().tolist(),
        ):
            score = float(confidence)
            if score < request.confidence:
                continue
            index = int(class_id)
            if model.handle.class_names is _BMD45_CLASSES and index == 0:
                continue
            detections.append(
                Detection(
                    bbox=[float(value) for value in box],
                    class_id=index,
                    label=model.handle.class_names.get(index, f"class {index}"),
                    confidence=score,
                )
            )
        return DetectionResult(detections=detections)

    def unload(self, model: LoadedModel) -> None:
        del model.handle.network
        del model.handle.postprocessor
        super().unload(model)
