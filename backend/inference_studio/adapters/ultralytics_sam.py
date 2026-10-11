"""SAM 2 and SAM 3 through Ultralytics.

Both families load Meta's released checkpoints (SAM 2 / 2.1 ``*.pt`` from facebookresearch/sam2 or the
Ultralytics mirrors, SAM 3 ``sam3.pt`` from the gated facebook/sam3 Hugging Face repo).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from inference_studio.results import SegmentationMask, SegmentationResult

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey
from .huggingface import parse_coordinates
from .sam import SamAdapter


def _overrides(key: ModelKey, imgsz: int) -> dict[str, Any]:
    return {
        "task": "segment",
        "mode": "predict",
        "imgsz": imgsz,
        "device": key.device,
        "half": key.precision == "fp16" and key.device.startswith("cuda"),
        "conf": 0.25,
        "save": False,
        "verbose": False,
    }


def _spatial_prompts(config: InferenceRequest, width: int, height: int) -> dict[str, Any]:
    if config.prompt_mode == "point":
        point = parse_coordinates(config.point_prompt, 2, "Point prompt")
        SamAdapter._check_bounds(point, width, height)
        return {"points": [point], "labels": [1 if config.point_foreground else 0]}
    if config.prompt_mode == "box":
        box = parse_coordinates(config.box_prompt, 4, "Box prompt")
        SamAdapter._check_bounds(box, width, height, box=True)
        return {"bboxes": [box]}
    raise ValueError("SAM requires a point or box prompt")


def to_segmentation(result: Any, config: InferenceRequest, semantic: bool = False) -> SegmentationResult:
    """Convert an Ultralytics ``Results`` into the studio's JSON-safe segmentation result."""

    values: list[SegmentationMask] = []
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy().astype(bool)
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.zeros(len(masks), int)
        for index, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names or []))
            label = str(names.get(int(classes[index]), "segment")) if semantic else f"segment {index + 1}"
            values.append(
                SegmentationMask(
                    class_id=int(classes[index]) if semantic else index,
                    label=label,
                    confidence=float(np.clip(scores[index], 0, 1)),
                    bbox=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    area=int(mask.sum()),
                    polygons=SamAdapter._polygons(mask),
                )
            )
    prompt = {"text": config.text_prompt, "point": config.point_prompt, "box": config.box_prompt}.get(config.prompt_mode, "")
    return SegmentationResult(masks=values, prompt_mode=config.prompt_mode, prompt=prompt)


@dataclass
class Sam2Handle:
    predictor: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


class Sam2Adapter(ModelAdapter):
    """Point/box segmentation with SAM 2 or SAM 2.1 (Hiera tiny, small, base-plus, large)."""

    name = "sam2"
    _sizes = {"tiny": "t", "small": "s", "base-plus": "b", "large": "l"}
    _versions = {"2.0": "sam2", "2.1": "sam2.1"}

    @classmethod
    def checkpoint_name(cls, version: str, size: str) -> str:
        return f"{cls._versions[version]}_{cls._sizes[size]}.pt"

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            from ultralytics.models.sam import SAM2Predictor
        except ImportError as exc:
            raise RuntimeError("SAM 2 requires the Ultralytics runtime") from exc

        # Ultralytics picks the architecture (and the SAM 2.1 variant) from the checkpoint file name,
        # so expose the stored upload under its canonical name for the duration of the build.
        staging = Path(tempfile.mkdtemp(prefix="vis-sam2-"))
        try:
            link = staging / self.checkpoint_name(key.version, key.size)
            link.symlink_to(weight_path.resolve())
            predictor = SAM2Predictor(overrides={**_overrides(key, 1024), "model": str(link)})
            try:
                predictor.setup_model(model=None, verbose=False)
            except RuntimeError as exc:  # strict state-dict load: wrong size or not a SAM 2 checkpoint
                raise RuntimeError(f"Checkpoint does not match SAM {key.version} {key.size}") from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return LoadedModel(handle=Sam2Handle(predictor), key=key, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray) -> dict[str, Any]:
        height, width = input_data.shape[:2]
        return {"image": input_data, "prompts": _spatial_prompts(request, width, height)}

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor: dict[str, Any]):
        with model.handle.lock:
            predictor = model.handle.predictor
            predictor.args.conf = request.confidence
            predictor.set_image(tensor["image"])
            try:
                return predictor(**tensor["prompts"])[0]
            finally:
                predictor.reset_image()

    def postprocess(self, model: LoadedModel, request: InferenceRequest, output) -> SegmentationResult:
        return to_segmentation(output, request)

    def unload(self, model: LoadedModel) -> None:
        del model.handle.predictor
        super().unload(model)


@dataclass
class Sam3Handle:
    weight_path: Path
    semantic: Any = None
    interactive: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class Sam3Adapter(ModelAdapter):
    """SAM 3: text prompts find every instance of a concept; point/box prompts segment one object."""

    name = "sam3"
    # Text (concept) prompts run at SAM 3's native 1008 px, as in Meta's reference processor. The point/box
    # predictor uses Ultralytics' default 644 px: at 840+ px its activations exceed a 6 GB GPU (3.7 GB peak at 644).
    semantic_imgsz = 1008
    interactive_imgsz = 644

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            import ultralytics.models.sam  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("SAM 3 requires the Ultralytics runtime") from exc
        handle = Sam3Handle(weight_path)
        return LoadedModel(handle=handle, key=key, weight_path=weight_path)

    def _predictor(self, model: LoadedModel, semantic: bool):
        handle: Sam3Handle = model.handle
        current = handle.semantic if semantic else handle.interactive
        if current is not None:
            return current
        # Each SAM 3 predictor holds its own ~1.7 GB (fp16) copy of the model; keep only one resident so
        # switching between text and point/box prompts fits on 6 GB GPUs.
        handle.semantic = handle.interactive = None
        self._release_cuda()
        try:
            import clip  # noqa: F401  Ultralytics would otherwise pip-install it at runtime
        except ImportError as exc:
            raise RuntimeError("SAM 3 text encoding requires: pip install git+https://github.com/ultralytics/CLIP.git") from exc
        from ultralytics.models.sam import SAM3Predictor, SAM3SemanticPredictor
        from ultralytics.models.sam.build_sam3 import build_interactive_sam3, build_sam3_image_model

        checkpoint = str(handle.weight_path)
        try:
            # Build explicitly: the Ultralytics predictor passes compile=False to torch.compile as a mode.
            network = build_sam3_image_model(checkpoint) if semantic else build_interactive_sam3(checkpoint, compile=None)
        except (RuntimeError, KeyError) as exc:
            raise RuntimeError("Checkpoint is not a SAM 3 (sam3.pt) checkpoint") from exc
        predictor = (SAM3SemanticPredictor if semantic else SAM3Predictor)(overrides={**_overrides(model.key, self.semantic_imgsz if semantic else self.interactive_imgsz), "model": checkpoint})
        predictor.setup_model(model=network, verbose=False)
        if semantic:
            handle.semantic = predictor
        else:
            handle.interactive = predictor
        return predictor

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray) -> dict[str, Any]:
        if request.prompt_mode == "text":
            phrases = [item.strip() for item in request.text_prompt.replace(";", ",").split(",") if item.strip()]
            if not phrases:
                raise ValueError("SAM 3 text mode requires at least one concept, e.g. 'wheel, rail'")
            return {"image": input_data, "semantic": True, "prompts": {"text": phrases}}
        height, width = input_data.shape[:2]
        return {"image": input_data, "semantic": False, "prompts": _spatial_prompts(request, width, height)}

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor: dict[str, Any]):
        with model.handle.lock:
            predictor = self._predictor(model, tensor["semantic"])
            predictor.args.conf = request.confidence
            predictor.set_image(tensor["image"])
            try:
                return predictor(**tensor["prompts"])[0], tensor["semantic"]
            finally:
                predictor.reset_image()

    def postprocess(self, model: LoadedModel, request: InferenceRequest, output) -> SegmentationResult:
        result, semantic = output
        return to_segmentation(result, request, semantic=semantic)

    @staticmethod
    def _release_cuda() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload(self, model: LoadedModel) -> None:
        model.handle.semantic = model.handle.interactive = None
        self._release_cuda()
        super().unload(model)
