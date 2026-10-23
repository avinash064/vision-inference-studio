from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib import metadata
from typing import Any

from inference_studio.runtime_diagnostics import classify_import_error


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
}
# Engines for individual model families or node packs. A failure is reported (and shown as that family's
# status) but never blocks the runtime: the rest of the studio must keep working, e.g. on a Windows machine
# where one native extension cannot load.
OPTIONAL_IMPORTS = {
    "groundingdino": "groundingdino",
    "clip": "clip",
    "chainner_ext": "chainner_ext",
    "paddleocr": "paddleocr",
    "paddle": "paddlepaddle",
    "pytesseract": "pytesseract",
    "faster_coco_eval": "faster-coco-eval",
}


# The "core" profile (API server only) is used by CI smoke tests of the packaged Windows app.
CORE_IMPORTS = ("fastapi", "uvicorn", "pydantic", "cv2", "numpy")


def inspect_runtime(profile: str = "full") -> dict[str, Any]:
    packages: dict[str, dict[str, str | bool]] = {}
    failures: list[str] = []
    modules: dict[str, Any] = {}
    optional_failures: list[str] = []
    required = {name: REQUIRED_IMPORTS[name] for name in CORE_IMPORTS} if profile == "core" else REQUIRED_IMPORTS
    optional = {} if profile == "core" else OPTIONAL_IMPORTS
    for module_name, distribution in {**required, **optional}.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                version = str(getattr(modules[module_name], "__version__", "unknown"))
            packages[module_name] = {"ok": True, "version": version}
        except Exception as exc:  # OSError covers native DLL/.so load failures
            packages[module_name] = {"ok": False, "optional": module_name in OPTIONAL_IMPORTS, "error": classify_import_error(exc)}
            (optional_failures if module_name in OPTIONAL_IMPORTS else failures).append(module_name)

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
        "optional_failures": optional_failures,
        "platform": platform.system().lower(),
        "architecture": platform.machine().lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Vision Inference Studio private runtime")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--profile", choices=["full", "core"], default="full")
    args = parser.parse_args()
    result = inspect_runtime(args.profile)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for name, status in result["packages"].items():
            label = "OK" if status["ok"] else ("OPTIONAL, UNAVAILABLE" if status.get("optional") else "FAILED")
            print(f"{name}: {label} {status.get('version', status.get('error', ''))}")
        print(f"CUDA: {'available' if result['cuda']['available'] else 'not available (CPU fallback)'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
