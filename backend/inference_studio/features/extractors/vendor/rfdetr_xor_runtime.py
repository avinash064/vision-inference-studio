"""Minimal runtime helpers for the isolated RF-DETR feature-XOR experiment."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


VERIFIED_NEGATIVE_FRAMES = tuple(range(0, 1001, 100))


def iter_dense_tensors(value: Any) -> Iterable[torch.Tensor]:
    """Yield dense feature tensors from RF-DETR's nested backbone output."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 4 and min(value.shape[-2:]) > 1:
            yield value
        return
    tensors = getattr(value, "tensors", None)
    if isinstance(tensors, torch.Tensor):
        yield from iter_dense_tensors(tensors)
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_dense_tensors(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_dense_tensors(child)


def discover_dense_levels(
    boundary_output: Any,
    input_hw: tuple[int, int],
) -> tuple[list[torch.Tensor], list[dict[str, object]]]:
    """Discover unique feature levels by shape, finest to coarsest."""
    candidates = list(iter_dense_tensors(boundary_output))
    if not candidates:
        raise RuntimeError("RF-DETR boundary hook produced no dense tensors")
    unique: dict[tuple[int, int], torch.Tensor] = {}
    multiplicity: Counter[tuple[int, int]] = Counter()
    for tensor in candidates:
        shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        multiplicity[shape] += 1
        unique.setdefault(shape, tensor)
    ordered = sorted(
        unique.values(),
        key=lambda tensor: int(tensor.shape[-2] * tensor.shape[-1]),
        reverse=True,
    )
    input_h, input_w = input_hw
    metadata: list[dict[str, object]] = []
    for index, tensor in enumerate(ordered):
        height, width = map(int, tensor.shape[-2:])
        metadata.append({
            "level_index": index,
            "shape": [height, width],
            "channels": int(tensor.shape[1]),
            "stride_y": input_h / height,
            "stride_x": input_w / width,
            "duplicate_shape_tensors_seen": multiplicity[(height, width)],
            "role": (
                "finest_rescue"
                if index == 0 and len(ordered) > 1
                else "primary"
            ),
        })
    return ordered, metadata


class RFDETRBoundaryFeatures:
    """Frozen RF-DETR backbone extractor with shape-based level discovery."""

    def __init__(
        self,
        weights: Path,
        device: torch.device,
        model_size: tuple[int, int],
    ) -> None:
        from rfdetr import RFDETRLargeDeprecated
        from rfdetr.utilities.tensors import NestedTensor

        self._nested_tensor = NestedTensor
        self.weights = weights.resolve()
        self.device = device
        self.model_size = model_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            wrapper = RFDETRLargeDeprecated(
                pretrain_weights=str(self.weights),
                device=str(device),
            )
        self.net = wrapper.model.model.to(device).eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.boundary = self.net.backbone
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device=device
        ).view(1, 3, 1, 1)
        self.level_metadata: list[dict[str, object]] | None = None

    def _preprocess(self, bgr: np.ndarray) -> Any:
        width, height = self.model_size
        rgb = cv2.cvtColor(
            cv2.resize(
                bgr, (width, height), interpolation=cv2.INTER_AREA
            ),
            cv2.COLOR_BGR2RGB,
        )
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).unsqueeze(0).float().div_(255.0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        mask = torch.zeros(
            (1, height, width), dtype=torch.bool, device=self.device
        )
        return self._nested_tensor(tensor, mask)

    def __call__(self, bgr: np.ndarray) -> list[torch.Tensor]:
        captured: list[Any] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured.append(output)

        handle = self.boundary.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self.boundary(self._preprocess(bgr))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                "expected one RF-DETR backbone hook call, "
                f"got {len(captured)}"
            )
        levels, metadata = discover_dense_levels(
            captured[0], (self.model_size[1], self.model_size[0])
        )
        if self.level_metadata is None:
            self.level_metadata = metadata
        elif [
            row["shape"] for row in metadata
        ] != [
            row["shape"] for row in self.level_metadata
        ]:
            raise RuntimeError("RF-DETR feature level shapes changed")
        return [level.detach() for level in levels]


def threshold_key(metadata: dict[str, object]) -> str:
    height, width = metadata["shape"]
    return f"{height}x{width}"


def fixed_calibration_threshold(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)].astype(np.float64)
    if finite.size == 0:
        raise ValueError("calibration contains no finite values")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    q995 = float(np.quantile(finite, 0.995))
    threshold = max(0.02, q995, median + 8.0 * 1.4826 * mad)
    return {
        "threshold": threshold,
        "median": median,
        "mad": mad,
        "q995": q995,
        "sample_count": int(finite.size),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reference_features_cached(
    extractor: RFDETRBoundaryFeatures,
    reference: np.ndarray,
    reference_path: Path,
    cache_dir: Path,
    registration: dict[str, object],
) -> tuple[list[torch.Tensor], Path, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_sha256": file_sha256(reference_path),
        "weights_name": extractor.weights.name,
        "weights_size": extractor.weights.stat().st_size,
        "registration": registration,
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:24]
    path = cache_dir / f"reference_features_{key}.pt"
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        features = [
            tensor.to(extractor.device) for tensor in cached["features"]
        ]
        extractor.level_metadata = cached["level_metadata"]
        return features, path, True
    features = extractor(reference)
    torch.save({
        "features": [feature.cpu() for feature in features],
        "level_metadata": extractor.level_metadata,
        "registration": registration,
    }, path)
    return features, path, False


def keep_components(
    raw_mask: np.ndarray,
    minimum_area: int,
) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (raw_mask > 0).astype(np.uint8), connectivity=8
    )
    output = np.zeros_like(raw_mask)
    kept = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            output[labels == label] = 255
            kept += 1
    return output, kept


def heat_panel(
    ratio: np.ndarray,
    label: str,
    size: tuple[int, int] = (640, 360),
) -> np.ndarray:
    visual = np.round(np.clip(ratio, 0.0, 2.0) * 127.5).astype(
        np.uint8
    )
    visual = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)
    visual = cv2.resize(visual, size, interpolation=cv2.INTER_LINEAR)
    cv2.putText(
        visual, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.60, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return visual


def polygons_from_mask(
    mask: np.ndarray,
    minimum_area: int,
    epsilon_fraction: float,
) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[np.ndarray] = []
    for contour in contours:
        if float(cv2.contourArea(contour)) < minimum_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(1.0, epsilon_fraction * perimeter)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        if len(polygon) >= 3:
            polygons.append(polygon)
    polygons.sort(key=cv2.contourArea, reverse=True)
    return polygons


def polygon_json(polygons: list[np.ndarray]) -> str:
    return json.dumps([
        polygon.reshape(-1, 2).astype(int).tolist()
        for polygon in polygons
    ])
