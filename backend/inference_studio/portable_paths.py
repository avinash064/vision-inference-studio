"""Cross-platform path rules for artifacts that must move between Windows, Linux, buckets and CVAT.

Exported files (manifest.jsonl, COCO, YOLO, canonical JSONL, promoted-recipe metadata) only ever contain
relative POSIX paths such as ``images/frame_000001.jpg``. The checks here recognise absolute paths of
*either* OS regardless of where the code runs, so a package written on Windows is validated correctly on
Linux and vice versa.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

# Characters Windows forbids in file names, plus ASCII control characters.
_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def is_absolute_any_platform(value: str) -> bool:
    """True for POSIX absolute paths, Windows drive paths (C:\\x, C:/x, C:x), and UNC paths (\\\\server\\share)."""
    if not value:
        return False
    if PurePosixPath(value).is_absolute():
        return True
    windows = PureWindowsPath(value)
    # A drive letter alone ("C:relative") is still bound to one machine, so treat it as non-portable too.
    return windows.is_absolute() or bool(windows.drive) or value.startswith(("\\\\", "//"))


def to_portable_relative(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` as a POSIX string, refusing anything that escapes ``root``."""
    relative = path.resolve().relative_to(root.resolve())
    text = relative.as_posix()
    if text.startswith("../") or text == ".." or is_absolute_any_platform(text):
        raise ValueError(f"Path is not portable relative to its package root: {text}")
    return text


def is_windows_safe_name(name: str) -> bool:
    """True when ``name`` is a single path component that Windows can create."""
    if not name or name in {".", ".."} or _WINDOWS_INVALID.search(name) or name.endswith((" ", ".")):
        return False
    return name.split(".")[0].upper() not in _WINDOWS_RESERVED


def safe_filename(name: str, replacement: str = "_", max_length: int = 120) -> str:
    """A file-name component valid on Windows and Linux: invalid characters replaced, reserved names escaped."""
    cleaned = _WINDOWS_INVALID.sub(replacement, name).strip(" .") or replacement
    stem, dot, suffix = cleaned.partition(".")
    if stem.upper() in _WINDOWS_RESERVED:
        cleaned = f"{stem}{replacement}{dot}{suffix}"
    if len(cleaned) > max_length:
        head, extension = (cleaned.rsplit(".", 1) + [""])[:2] if "." in cleaned[-12:] else (cleaned, "")
        cleaned = head[: max_length - len(extension) - 1] + (f".{extension}" if extension else "")
    return cleaned


def windows_unsafe_member(member: str) -> str | None:
    """The first path component of an archive member that Windows cannot create, or None."""
    for part in PurePosixPath(member.replace("\\", "/")).parts:
        if part not in {"/", ""} and not is_windows_safe_name(part):
            return part
    return None
