from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


class IntegrityError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(root: Path) -> tuple[str, int]:
    digest, total = hashlib.sha256(), 0
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        digest.update(sha256_file(path).encode("ascii"))
        total += size
    return digest.hexdigest(), total


def _safe_name(name: str) -> Path:
    normalized = name.replace("\\", "/")
    value = PurePosixPath(normalized)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise IntegrityError(f"Unsafe archive member: {name}")
    return Path(*value.parts)


def safe_extract_archive(archive: Path, destination: Path, max_total_bytes: int, max_file_bytes: int) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    relative = _safe_name(member.filename)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise IntegrityError(f"Archive symlink is forbidden: {member.filename}")
                    if member.file_size > max_file_bytes:
                        raise IntegrityError(f"Archive member exceeds configured limit: {member.filename}")
                    total += member.file_size
                    if total > max_total_bytes:
                        raise IntegrityError("Archive exceeds configured extraction limit")
                    target = (destination / relative).resolve()
                    if destination.resolve() not in target.parents and target != destination.resolve():
                        raise IntegrityError(f"Archive member escapes destination: {member.filename}")
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with package.open(member) as source, target.open("xb") as output:
                            shutil.copyfileobj(source, output, 1024 * 1024)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as package:
                members = package.getmembers()
                for member in members:
                    relative = _safe_name(member.name)
                    if member.issym() or member.islnk() or member.isdev():
                        raise IntegrityError(f"Archive link/device is forbidden: {member.name}")
                    if member.size > max_file_bytes:
                        raise IntegrityError(f"Archive member exceeds configured limit: {member.name}")
                    total += member.size
                    if total > max_total_bytes:
                        raise IntegrityError("Archive exceeds configured extraction limit")
                    target = (destination / relative).resolve()
                    if destination.resolve() not in target.parents and target != destination.resolve():
                        raise IntegrityError(f"Archive member escapes destination: {member.name}")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = package.extractfile(member)
                        if source is None:
                            raise IntegrityError(f"Unreadable archive member: {member.name}")
                        with source, target.open("xb") as output:
                            shutil.copyfileobj(source, output, 1024 * 1024)
        else:
            raise IntegrityError("Downloaded file is not a valid ZIP or TAR archive")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def ensure_cache_capacity(root: Path, incoming_bytes: int, maximum_bytes: int, protected: set[Path] | None = None) -> None:
    protected = {path.resolve() for path in (protected or set())}
    current = directory_size(root)
    if current + incoming_bytes <= maximum_bytes:
        return
    candidates = sorted((p for p in root.iterdir() if p.resolve() not in protected), key=lambda p: p.stat().st_atime)
    for candidate in candidates:
        size = directory_size(candidate) if candidate.is_dir() else candidate.stat().st_size
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink(missing_ok=True)
        current -= size
        if current + incoming_bytes <= maximum_bytes:
            return
    raise IntegrityError("Public Hub cache size limit exceeded")
