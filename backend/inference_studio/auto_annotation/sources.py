from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable

import cv2

from inference_studio.assets import AssetRecord, AssetStore
from inference_studio.video import VideoFrame, VideoReader

from .sampling import FrameSampler
from .schemas import FrameSamplingPolicy, SourceConfig, SourceType


class SourceAccessError(ValueError):
    pass


class SourceService:
    def __init__(
        self,
        assets: AssetStore,
        input_roots: Iterable[Path],
        preview_root: Path,
        approvals_file: Path | None = None,
    ) -> None:
        self.assets = assets
        self.input_roots = [Path(root).expanduser().resolve() for root in input_roots]
        self.preview_root = preview_root.resolve()
        self.preview_root.mkdir(parents=True, exist_ok=True)
        # Desktop mode: files/folders the user picked in the native OS dialog (approved by the desktop shell).
        self.approvals_file = approvals_file
        self.approved = self._load_approvals()

    def _load_approvals(self) -> list[Path]:
        if self.approvals_file is None:
            return []
        try:
            return [Path(value) for value in json.loads(self.approvals_file.read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError):
            return []

    def approve(self, raw: str) -> Path:
        """Allow a user-selected file or folder (desktop shell only) and remember it across restarts."""
        if self.approvals_file is None:
            raise SourceAccessError("Path approval is only available in desktop mode")
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise SourceAccessError("Selected path does not exist")
        if not any(path == root or root in path.parents for root in [*self.input_roots, *self.approved]):
            self.approved.append(path)
            self.approvals_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.approvals_file.with_suffix(".tmp")
            temporary.write_text(json.dumps([str(item) for item in self.approved], indent=2), encoding="utf-8")
            os.replace(temporary, self.approvals_file)
        return path

    def _allowed_path(self, raw: str, *, directory: bool) -> Path:
        path = Path(raw).expanduser().resolve()
        if not any(path == root or root in path.parents for root in [*self.input_roots, *self.approved]):
            raise SourceAccessError(
                "Source path is outside VIS_AUTO_ANNOTATION_INPUT_ROOTS"
            )
        if directory and not path.is_dir():
            raise SourceAccessError("Image source folder does not exist")
        if not directory and not path.is_file():
            raise SourceAccessError("Video source file does not exist")
        return path

    def video_record(self, config: SourceConfig) -> AssetRecord | None:
        if not config.asset_id:
            return None
        record = self.assets.get(config.asset_id)
        if record.kind != "video":
            raise SourceAccessError("Selected asset is not a video")
        return record

    def video_path(self, config: SourceConfig) -> tuple[Path, AssetRecord | None]:
        record = self.video_record(config)
        if record is not None:
            return Path(record.path).resolve(), record
        return self._allowed_path(str(config.path), directory=False), None

    def image_inventory(self, config: SourceConfig) -> tuple[Path, list[dict]]:
        root = self._allowed_path(str(config.path), directory=True)
        glob = root.rglob("*") if config.recursive else root.glob("*")
        paths = sorted(
            path
            for path in glob
            if path.is_file() and path.suffix.lower() in AssetStore.IMAGE_EXTENSIONS
        )
        values = []
        for path in paths:
            relative = path.relative_to(root).as_posix()
            resolved = path.resolve()
            if root not in resolved.parents:
                raise SourceAccessError(
                    f"Image source entry escapes its approved root: {relative}"
                )
            image = cv2.imread(str(resolved))
            sha256 = self.file_sha256(resolved)
            if image is None:
                values.append(
                    {
                        "item_id": self.image_item_id(relative),
                        "relative_path": relative,
                        "path": str(resolved),
                        "size": resolved.stat().st_size,
                        "sha256": sha256,
                        "width": 0,
                        "height": 0,
                        "error": "Image cannot be decoded",
                    }
                )
                continue
            height, width = image.shape[:2]
            values.append(
                {
                    "item_id": self.image_item_id(relative),
                    "relative_path": relative,
                    "path": str(resolved),
                    "size": resolved.stat().st_size,
                    "sha256": sha256,
                    "width": width,
                    "height": height,
                }
            )
        return root, values

    @staticmethod
    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def inventory_sha256(items: list[dict]) -> str:
        identity = [
            {
                "relative_path": item["relative_path"],
                "size": item["size"],
                "sha256": item["sha256"],
            }
            for item in items
        ]
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def image_item_id(relative_path: str) -> str:
        return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]

    def inspect(self, source_type: SourceType, config: SourceConfig) -> dict:
        if source_type == SourceType.IMAGE_FOLDER:
            root, items = self.image_inventory(config)
            readable = [item for item in items if not item.get("error")]
            return {
                "source_type": source_type.value,
                "folder_name": root.name,
                "source_count": len(items),
                "readable_count": len(readable),
                "failed_count": len(items) - len(readable),
                "recursive": config.recursive,
                "snapshot_sha256": self.inventory_sha256(items),
                "items": [
                    {key: value for key, value in item.items() if key != "path"}
                    for item in items[:200]
                ],
                "truncated": len(items) > 200,
            }
        path, record = self.video_path(config)
        with VideoReader(path) as reader:
            return {
                "source_type": source_type.value,
                "asset_id": record.asset_id if record else None,
                "filename": record.filename if record else path.name,
                "codec": record.codec if record else reader.codec,
                "width": reader.width,
                "height": reader.height,
                "duration_seconds": reader.duration_seconds,
                "nominal_fps": reader.fps,
                "total_frames": reader.frame_count,
                "time_base": reader.time_base,
                "snapshot_sha256": record.sha256 if record else self.file_sha256(path),
            }

    def preview_sampling(
        self, config: SourceConfig, policy: FrameSamplingPolicy, limit: int
    ) -> dict:
        path, record = self.video_path(config)
        request_id = f"aap_{uuid.uuid4().hex}"
        target = self.preview_root / request_id
        target.mkdir(mode=0o700)
        previews = []
        with VideoReader(path) as reader:
            estimated = FrameSampler.estimated_count(
                policy, reader.fps, reader.frame_count, reader.duration_seconds
            )
            sampler = FrameSampler(policy, reader.fps)
            sample_count = min(limit, estimated)
            target_ordinals = (
                {
                    round(index * (estimated - 1) / max(sample_count - 1, 1))
                    for index in range(sample_count)
                }
                if estimated
                else set()
            )
            selected_ordinal = 0
            for frame_index in range(reader.frame_count):
                if sampler.select(frame_index, frame_index / reader.fps):
                    if selected_ordinal not in target_ordinals:
                        selected_ordinal += 1
                        continue
                    reader.seek(frame_index)
                    packet = next(reader.frames(), None)
                    if packet is None:
                        selected_ordinal += 1
                        continue
                    filename = f"frame_{packet.index:012d}.jpg"
                    output = target / filename
                    if not cv2.imwrite(
                        str(output), packet.image, [cv2.IMWRITE_JPEG_QUALITY, 82]
                    ):
                        selected_ordinal += 1
                        continue
                    previews.append(
                        self._frame_public(
                            packet, f"/auto-annotation-previews/{request_id}/{filename}"
                        )
                    )
                    selected_ordinal += 1
            metadata = {
                "filename": record.filename if record else path.name,
                "fps": reader.fps,
                "frame_count": reader.frame_count,
                "duration_seconds": reader.duration_seconds,
                "width": reader.width,
                "height": reader.height,
            }
        self._prune_previews()
        return {
            "request_id": request_id,
            "estimated_sampled_frames": estimated,
            "source": metadata,
            "frames": previews,
        }

    @staticmethod
    def _frame_public(packet: VideoFrame, url: str | None = None) -> dict:
        return {
            "frame_index": packet.index,
            "pts": packet.pts,
            "time_base": packet.time_base,
            "timestamp_seconds": packet.timestamp_seconds,
            "width": packet.image.shape[1],
            "height": packet.image.shape[0],
            "url": url,
        }

    def preview_images(
        self, config: SourceConfig, limit: int
    ) -> list[tuple[dict, object]]:
        _, items = self.image_inventory(config)
        readable = [item for item in items if not item.get("error")]
        if not readable:
            return []
        indexes = sorted(
            {
                round(index * (len(readable) - 1) / max(limit - 1, 1))
                for index in range(min(limit, len(readable)))
            }
        )
        return [
            (readable[index], cv2.imread(readable[index]["path"])) for index in indexes
        ]

    def preview_video_frames(
        self, config: SourceConfig, policy: FrameSamplingPolicy, limit: int
    ) -> list[tuple[dict, object]]:
        path, record = self.video_path(config)
        values = []
        with VideoReader(path) as reader:
            sampler = FrameSampler(policy, reader.fps)
            for packet in reader.frames():
                if sampler.select(packet.index, packet.timestamp_seconds):
                    values.append(
                        (
                            {
                                **self._frame_public(packet),
                                "source_video_id": record.asset_id
                                if record
                                else hashlib.sha256(path.name.encode()).hexdigest()[
                                    :24
                                ],
                                "source_video_name": record.filename
                                if record
                                else path.name,
                            },
                            packet.image.copy(),
                        )
                    )
                    if len(values) >= limit:
                        break
        return values

    def _prune_previews(self) -> None:
        directories = sorted(
            (path for path in self.preview_root.glob("aap_*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in directories[5:]:
            shutil.rmtree(path, ignore_errors=True)
