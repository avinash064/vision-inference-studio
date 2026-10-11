from __future__ import annotations

import json
from typing import Any

import numpy as np

from inference_studio.runtime import ImageValue, TensorValue

from .image import ChainnerImageAdapter


class ChainnerValueAdapter:
    """Metadata-driven bridge between chaiNNer ports and runtime values."""

    @staticmethod
    def is_image(definition: Any) -> bool:
        expression = getattr(definition, "output_type", getattr(definition, "input_type", None))
        return "image" in json.dumps(expression, default=str).lower()

    def to_chainner(self, input_definition: Any, runtime_value: Any) -> Any:
        if isinstance(runtime_value, ImageValue):
            if not self.is_image(input_definition):
                raise ValueError("chainner_value_type_mismatch: ImageValue connected to a non-image chaiNNer input")
            return ChainnerImageAdapter.to_chainner(runtime_value.image)
        if isinstance(runtime_value, TensorValue):
            if self.is_image(input_definition):
                raise ValueError("chainner_value_type_mismatch: TensorValue connected to an image chaiNNer input")
            return runtime_value.value
        return runtime_value

    def to_vision(self, output_definition: Any, raw_value: Any) -> Any:
        if isinstance(raw_value, np.ndarray) and self.is_image(output_definition):
            return ImageValue(raw_value)
        if isinstance(raw_value, np.ndarray):
            device = str(getattr(raw_value, "device", "cpu"))
            return TensorValue(raw_value, tuple(raw_value.shape), str(raw_value.dtype), device)
        return raw_value
