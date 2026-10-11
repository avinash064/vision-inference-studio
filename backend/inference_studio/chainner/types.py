from __future__ import annotations

import json
import re
from typing import Any


class ChainnerTypeAdapter:
    """Conservatively bridge Navi/chaiNNer types into Vision port types."""

    _names = {
        "image": "image",
        "color": "color",
        "number": "number",
        "int": "number",
        "uint": "number",
        "bool": "boolean",
        "string": "string",
        "directory": "directory",
        "file": "file",
        "pytorchmodel": "pytorch_model",
        "onnxmodel": "onnx_model",
        "ncnnmodel": "ncnn_model",
        "tensorrtengine": "tensorrt_engine",
        "model": "model",
        "sequence": "sequence",
        "tensor": "tensor",
        "matrix": "tensor",
    }

    @classmethod
    def adapt(cls, expression: Any, associated_type: Any = None) -> list[str]:
        text = json.dumps(expression, default=str).lower().replace("_", "")
        tokens = set(re.findall(r"[a-z][a-z0-9]*", text))
        result: list[str] = []
        for name, mapped in cls._names.items():
            if name in tokens and mapped not in result:
                result.append(mapped)
        if result:
            return result
        if associated_type is not None:
            associated = getattr(associated_type, "__name__", str(associated_type)).lower().replace("_", "")
            if associated == "image":
                return ["image"]
            if associated == "ndarray":
                return ["tensor"]
            if associated in {"int", "float", "seed"}:
                return ["number"]
            if associated in {"bool", "str", "path"}:
                return [{"bool": "boolean", "str": "string", "path": "file"}[associated]]
            for name, mapped in cls._names.items():
                if name in associated:
                    return [mapped]
        return ["object"]
