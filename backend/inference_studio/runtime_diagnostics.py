"""Runtime and model-family capability diagnostics (``GET /api/v1/runtime/diagnostics``).

Model adapters keep importing their runtimes lazily; this module probes them separately so one missing or
broken dependency (for example a Windows DLL that fails to load) is reported per family instead of
preventing startup. Nothing here returns secrets or environment values.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import threading
from pathlib import Path
from typing import Any

from inference_studio.binaries import resolve_binary

AVAILABLE, DEPENDENCY_MISSING = "AVAILABLE", "DEPENDENCY_MISSING"
UNSUPPORTED_PLATFORM, CONFIGURATION_REQUIRED = "UNSUPPORTED_PLATFORM", "CONFIGURATION_REQUIRED"

# Python modules each adapter imports when a model loads.
_ADAPTER_MODULES: dict[str, tuple[str, ...]] = {
    "yolo": ("torch", "ultralytics"), "yolo-seg": ("torch", "ultralytics"), "rtdetr": ("torch", "ultralytics"),
    "rtdetrv2": ("torch", "torchvision"), "rfdetr": ("torch", "rfdetr"), "dinov3": ("torch", "transformers"),
    "grounding-dino": ("torch", "groundingdino"), "sam": ("torch", "transformers"), "sam2": ("torch", "ultralytics"),
    "sam3": ("torch", "ultralytics", "clip"), "paddleocr": ("paddle", "paddleocr"), "tesseract": ("pytesseract",),
    "vllm": ("httpx",), "onnx": ("onnxruntime",),
}
# Adapters known not to work on a platform. Empty until a real failure is confirmed; never guessed.
_UNSUPPORTED: dict[str, set[str]] = {"windows": set(), "linux": set(), "darwin": set()}

_lock = threading.Lock()
_module_cache: dict[str, dict[str, Any]] = {}


def platform_name() -> str:
    return {"win32": "windows", "linux": "linux", "darwin": "darwin"}.get(sys.platform, sys.platform)


def architecture() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)


def classify_import_error(exc: BaseException) -> str:
    """A short, actionable explanation of why an import failed."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "dll load failed" in lowered or "winerror 126" in lowered or "winerror 127" in lowered:
        return f"Native library failed to load ({text}). Reinstall the runtime; on Windows also install the latest Microsoft Visual C++ Redistributable."
    if isinstance(exc, ModuleNotFoundError):
        return f"Python package not installed ({exc.name or text})"
    if "cuda" in lowered:
        return f"CUDA library unavailable ({text})"
    return text


def probe_module(name: str) -> dict[str, Any]:
    """Import ``name`` once (cached) and record its version or the classified failure."""
    with _lock:
        if name in _module_cache:
            return _module_cache[name]
    try:
        module = importlib.import_module(name)
        result: dict[str, Any] = {"ok": True, "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:  # includes OSError from native extensions
        result = {"ok": False, "error": classify_import_error(exc)}
    with _lock:
        _module_cache[name] = result
    return result


def family_status(adapter: str) -> dict[str, Any]:
    if adapter in _UNSUPPORTED.get(platform_name(), set()):
        return {"status": UNSUPPORTED_PLATFORM, "detail": f"Not supported on {platform_name()}"}
    for module in _ADAPTER_MODULES.get(adapter, ()):
        probe = probe_module(module)
        if not probe["ok"]:
            return {"status": DEPENDENCY_MISSING, "detail": f"{module}: {probe['error']}"}
    if adapter == "tesseract" and resolve_binary("tesseract") is None:
        return {"status": CONFIGURATION_REQUIRED, "detail": "Tesseract OCR engine not found. Install it, or set VIS_TESSERACT_CMD to tesseract executable."}
    return {"status": AVAILABLE, "detail": ""}


def cuda_report() -> dict[str, Any]:
    torch_probe = probe_module("torch")
    report: dict[str, Any] = {"torch": torch_probe.get("version"), "cuda_available": False, "cuda_runtime": None, "gpu": None, "cuda_status": "CUDA UNAVAILABLE", "cuda_detail": ""}
    if not torch_probe["ok"]:
        report["cuda_detail"] = f"PyTorch unavailable: {torch_probe['error']}"
        return report
    import torch

    report["cuda_runtime"] = torch.version.cuda
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        available, report["cuda_detail"] = False, classify_import_error(exc)
    report["cuda_available"] = available
    if available:
        properties = torch.cuda.get_device_properties(0)
        report.update(cuda_status="CUDA READY", gpu={"name": properties.name, "vram_bytes": int(properties.total_memory), "count": torch.cuda.device_count()})
    elif not report["cuda_detail"]:
        report["cuda_detail"] = (
            "This PyTorch build has no CUDA support (CPU-only wheel)." if not torch.version.cuda
            else "No usable NVIDIA GPU/driver detected; CPU-compatible model families remain available."
        )
    return report


def runtime_manifest() -> dict[str, Any] | None:
    runtime_dir = os.getenv("VIS_RUNTIME_DIR")
    if not runtime_dir:
        return None
    try:
        return json.loads((Path(runtime_dir) / "runtime-manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def runtime_diagnostics(families: dict[str, Any]) -> dict[str, Any]:
    source = os.getenv("VIS_RUNTIME_SOURCE", "development")
    manifest = runtime_manifest()
    return {
        "platform": platform_name(),
        "architecture": architecture(),
        "python": platform.python_version(),
        "runtime_source": source,
        # The interpreter path is shown only for runtimes the user chose; managed runtimes are an implementation detail.
        "python_executable": sys.executable if source in {"configured", "env_override", "development", "system"} else None,
        "runtime_manifest": {key: manifest.get(key) for key in ("platform", "arch", "python", "application_version", "runtime_version", "torch", "cuda_runtime")} if manifest else None,
        **cuda_report(),
        "binaries": {name: resolve_binary(name) is not None for name in ("tesseract", "ffmpeg")},
        "families": {name: {"display_name": family.display_name, **family_status(family.adapter)} for name, family in families.items()},
    }
