from __future__ import annotations

from pathlib import Path
from typing import Any


def apply_uploaded_state_dict(network: Any, weight_path: Path) -> None:
    """Apply a local, data-only checkpoint to an initialized HF model."""

    import torch

    if weight_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load this checkpoint") from exc
        state: Any = load_file(str(weight_path), device="cpu")
    else:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)

    while isinstance(state, dict) and any(key in state for key in ("state_dict", "model")):
        state = state.get("state_dict", state.get("model"))
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Checkpoint does not contain a model state dictionary")

    if all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    incompatible = network.load_state_dict(state, strict=False)
    matched = len(state) - len(incompatible.unexpected_keys)
    if matched <= 0:
        raise RuntimeError("Checkpoint does not contain weights compatible with the selected architecture")


def move_inputs(inputs: Any, device: str, half: bool) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if hasattr(value, "to"):
            value = value.to(device)
            if half and isinstance(value, torch.Tensor) and value.is_floating_point():
                value = value.half()
        moved[key] = value
    return moved


def to_numpy(value: Any):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def parse_coordinates(value: str, expected: int, label: str) -> list[float]:
    try:
        coordinates = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{label} must contain comma-separated numbers") from exc
    if len(coordinates) != expected:
        raise ValueError(f"{label} requires {expected} comma-separated coordinates")
    return coordinates
