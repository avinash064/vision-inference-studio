from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference_studio.results import SegmentationMask, SegmentationResult

from .base import LoadedModel, ModelAdapter, ModelConfig
from .huggingface import apply_uploaded_state_dict, move_inputs, parse_coordinates, to_numpy


@dataclass
class SamHandle:
    processor: Any
    network: Any


class SamAdapter(ModelAdapter):
    name = "sam"
    _models = {
        "vit-b": "facebook/sam-vit-base",
        "vit-l": "facebook/sam-vit-large",
        "vit-h": "facebook/sam-vit-huge",
    }

    def load(self, config: ModelConfig, weight_path: Path) -> LoadedModel:
        try:
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise RuntimeError("SAM requires the Transformers runtime") from exc

        identifier = self._models[config.size]
        processor = SamProcessor.from_pretrained(identifier)
        network = SamModel.from_pretrained(identifier)
        apply_uploaded_state_dict(network, weight_path)
        network.to(config.device).eval()
        if config.precision == "fp16":
            network.half()
        return LoadedModel(handle=SamHandle(processor, network), config=config, weight_path=weight_path)

    @staticmethod
    def _check_bounds(values: list[float], width: int, height: int, box: bool = False) -> None:
        points = list(zip(values[0::2], values[1::2]))
        if any(x < 0 or y < 0 or x > width or y > height for x, y in points):
            raise ValueError("SAM prompt coordinates must be inside the source image")
        if box and (values[2] <= values[0] or values[3] <= values[1]):
            raise ValueError("SAM box prompt must have x2 > x1 and y2 > y1")

    def preprocess(self, model: LoadedModel, input_data: np.ndarray) -> dict[str, Any]:
        rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        height, width = input_data.shape[:2]
        prompt_kwargs: dict[str, Any]
        if model.config.prompt_mode == "point":
            point = parse_coordinates(model.config.point_prompt, 2, "Point prompt")
            self._check_bounds(point, width, height)
            prompt_kwargs = {
                "input_points": [[point]],
                "input_labels": [[1 if model.config.point_foreground else 0]],
            }
        elif model.config.prompt_mode == "box":
            box = parse_coordinates(model.config.box_prompt, 4, "Box prompt")
            self._check_bounds(box, width, height, box=True)
            prompt_kwargs = {"input_boxes": [[box]]}
        else:
            raise ValueError("SAM requires point or box prompt mode")

        encoded = model.handle.processor(images=rgb, return_tensors="pt", **prompt_kwargs)
        return move_inputs(encoded, model.config.device, model.config.precision == "fp16")

    def infer(self, model: LoadedModel, tensor: dict[str, Any]):
        import torch

        with torch.inference_mode():
            output = model.handle.network(**tensor, multimask_output=model.config.multimask)
        return output, tensor

    @staticmethod
    def _polygons(mask: np.ndarray) -> list[list[float]]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for contour in contours:
            if cv2.contourArea(contour) < 4:
                continue
            epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
            points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(points) >= 3:
                polygons.append(points.astype(float).reshape(-1).tolist())
        return polygons

    def postprocess(self, model: LoadedModel, output) -> SegmentationResult:
        raw, inputs = output
        processed = model.handle.processor.post_process_masks(
            raw.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
            mask_threshold=model.config.mask_threshold,
            binarize=True,
        )
        masks = np.asarray(to_numpy(processed[0]))
        scores = np.asarray(to_numpy(raw.iou_scores), dtype=float)[0]
        if masks.ndim == 2:
            masks = masks[None, None, ...]
        elif masks.ndim == 3:
            masks = masks[None, ...]
        if scores.ndim == 1:
            scores = scores[None, ...]

        values: list[SegmentationMask] = []
        for index, candidates in enumerate(masks):
            candidate_scores = scores[min(index, len(scores) - 1)]
            selected = int(np.argmax(candidate_scores))
            mask = np.asarray(candidates[selected], dtype=bool)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            values.append(
                SegmentationMask(
                    class_id=index,
                    label=f"segment {index + 1}",
                    confidence=float(np.clip(candidate_scores[selected], 0, 1)),
                    bbox=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    area=int(mask.sum()),
                    polygons=self._polygons(mask),
                )
            )

        prompt = model.config.point_prompt if model.config.prompt_mode == "point" else model.config.box_prompt
        return SegmentationResult(masks=values, prompt_mode=model.config.prompt_mode, prompt=prompt)

    def unload(self, model: LoadedModel) -> None:
        del model.handle.network
        super().unload(model)
