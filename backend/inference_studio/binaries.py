"""Locate optional external executables (Tesseract, FFmpeg) without requiring them on PATH.

Resolution order: explicit environment override, binaries shipped next to the packaged app
(``<VIS_PROJECT_ROOT>/bin``), PATH, then the standard install locations of the official Windows installers.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_ENV_OVERRIDES = {"tesseract": ("VIS_TESSERACT_CMD", "TESSERACT_CMD"), "ffmpeg": ("VIS_FFMPEG", "FFMPEG_BINARY"), "ffprobe": ("VIS_FFPROBE",)}
_WINDOWS_DEFAULTS = {
    "tesseract": [r"%ProgramFiles%\Tesseract-OCR\tesseract.exe", r"%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe", r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"],
}


def resolve_binary(name: str) -> str | None:
    executable = f"{name}.exe" if os.name == "nt" else name
    for variable in _ENV_OVERRIDES.get(name, ()):
        value = os.getenv(variable, "").strip()
        if value and Path(value).is_file():
            return value
    root = os.getenv("VIS_PROJECT_ROOT")
    if root and (Path(root) / "bin" / executable).is_file():
        return str(Path(root) / "bin" / executable)
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        for candidate in _WINDOWS_DEFAULTS.get(name, []):
            path = Path(os.path.expandvars(candidate))
            if path.is_file():
                return str(path)
    return None
