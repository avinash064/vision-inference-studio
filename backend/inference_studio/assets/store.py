from __future__ import annotations

import hashlib
import json
import mimetypes
import uuid
from pathlib import Path
from typing import BinaryIO, Literal

import cv2
from pydantic import BaseModel, Field


class AssetError(ValueError):
    pass


class AssetRecord(BaseModel):
    asset_id: str
    kind: Literal["image", "video"]
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    width: int = 0
    height: int = 0
    frame_count: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    codec: str | None = None
    path: str = Field(exclude=True)


class AssetStore:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, asset_id: str) -> Path:
        if not asset_id.startswith("ast_") or not asset_id[4:].isalnum():
            raise AssetError("Invalid asset id")
        return self.root / f"{asset_id}.json"

    def save(self, source: BinaryIO, filename: str, expected_kind: Literal["image", "video"]) -> AssetRecord:
        extension = Path(filename).suffix.lower()
        allowed = self.IMAGE_EXTENSIONS if expected_kind == "image" else self.VIDEO_EXTENSIONS
        if extension not in allowed:
            raise AssetError(f"Unsupported {expected_kind} extension: {extension or '(none)'}")
        asset_id = f"ast_{uuid.uuid4().hex}"
        target = self.root / f"{asset_id}{extension}"
        digest = hashlib.sha256()
        total = 0
        try:
            with target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise AssetError("Upload exceeds configured size limit")
                    digest.update(chunk)
                    output.write(chunk)
            record = self._inspect(target, asset_id, Path(filename).name, expected_kind, total, digest.hexdigest())
            self._metadata_path(asset_id).write_text(record.model_dump_json(indent=2), encoding="utf-8")
            return record
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def import_path(self, source_path: str | Path, expected_kind: Literal["image", "video"] = "image") -> AssetRecord:
        """Copy a trusted server-side file into the existing runtime asset store."""
        path = Path(source_path).resolve()
        if not path.is_file():
            raise AssetError("Source asset does not exist")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for metadata in self.root.glob("ast_*.json"):
            try:
                data = json.loads(metadata.read_text(encoding="utf-8"))
                if data.get("sha256") == digest and data.get("kind") == expected_kind:
                    return self.get(data["asset_id"])
            except (OSError, json.JSONDecodeError, AssetError):
                continue
        with path.open("rb") as source:
            return self.save(source, path.name, expected_kind)

    def _inspect(self, path: Path, asset_id: str, filename: str, kind: Literal["image", "video"], size: int, sha256: str) -> AssetRecord:
        if kind == "image":
            image = cv2.imread(str(path))
            if image is None:
                raise AssetError("OpenCV could not decode the uploaded image")
            height, width = image.shape[:2]
            return AssetRecord(asset_id=asset_id, kind=kind, filename=filename, content_type=mimetypes.guess_type(filename)[0] or "application/octet-stream", size_bytes=size, sha256=sha256, width=width, height=height, path=str(path))
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise AssetError("OpenCV could not open the uploaded video")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ") or "unknown"
        readable, _ = capture.read()
        capture.release()
        if width <= 0 or height <= 0 or frames <= 0 or fps <= 0 or not readable:
            raise AssetError("Uploaded video has no readable frames or valid stream metadata")
        return AssetRecord(asset_id=asset_id, kind=kind, filename=filename, content_type=mimetypes.guess_type(filename)[0] or "application/octet-stream", size_bytes=size, sha256=sha256, width=width, height=height, frame_count=frames, fps=fps, duration_seconds=frames / fps, codec=codec, path=str(path))

    def get(self, asset_id: str) -> AssetRecord:
        metadata = self._metadata_path(asset_id)
        if not metadata.is_file():
            raise AssetError(f"Unknown asset: {asset_id}")
        data = json.loads(metadata.read_text(encoding="utf-8"))
        data["path"] = str(self.root / f"{asset_id}{Path(data['filename']).suffix.lower()}")
        record = AssetRecord.model_validate(data)
        if not Path(record.path).is_file() or Path(record.path).resolve().parent != self.root:
            raise AssetError("Asset data is missing")
        return record
