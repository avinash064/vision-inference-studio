from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib import metadata
from typing import Any


REQUIRED_IMPORTS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "rfdetr": "rfdetr",
    "onnxruntime": "onnxruntime-gpu",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "groundingdino": "groundingdino",
    "clip": "clip",
    "chainner_ext": "chainner_ext",
    "paddleocr": "paddleocr",
    "paddle": "paddlepaddle",
    "pytesseract": "pytesseract",
    "faster_coco_eval": "faster-coco-eval",
}


def inspect_runtime() -> dict[str, Any]:
    packages: dict[str, dict[str, str | bool]] = {}
    failures: list[str] = []
    modules: dict[str, Any] = {}
    for module_name, distribution in REQUIRED_IMPORTS.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                version = str(getattr(modules[module_name], "__version__", "unknown"))
            packages[module_name] = {"ok": True, "version": version}
        except Exception as exc:
            packages[module_name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(module_name)

    cuda: dict[str, Any] = {"available": False, "version": None, "devices": []}
    torch = modules.get("torch")
    if torch is not None:
        try:
            cuda["available"] = bool(torch.cuda.is_available())
            cuda["version"] = torch.version.cuda
            if cuda["available"]:
                cuda["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        except Exception as exc:
            cuda["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "ok": not failures,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": packages,
        "cuda": cuda,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Vision Inference Studio private runtime")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    result = inspect_runtime()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for name, status in result["packages"].items():
            print(f"{name}: {'OK' if status['ok'] else 'FAILED'} {status.get('version', status.get('error', ''))}")
        print(f"CUDA: {'available' if result['cuda']['available'] else 'not available (CPU fallback)'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
