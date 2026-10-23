from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator

import cv2
import numpy as np


class VideoReaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_seconds: float
    image: np.ndarray
    decode_ms: float
    pts: int | None = None
    time_base: str | None = None


class VideoReader:
    """Small sequential OpenCV decoder boundary, replaceable by PyAV/FFmpeg."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.capture: cv2.VideoCapture | None = None
        self.fps = 0.0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.codec = "unknown"
        self.duration_seconds = 0.0
        self.time_base: str | None = None
        self.next_index = 0

    def open(self) -> "VideoReader":
        self.close()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoReaderError("OpenCV could not open the video")
        self.capture = capture
        self.fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ") or "unknown"
        self.next_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if self.fps <= 0 or self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            self.close()
            raise VideoReaderError("Video has invalid or empty stream metadata")
        self.duration_seconds = self.frame_count / self.fps
        self.time_base = "1/1000"
        return self

    def frames(self) -> Iterator[VideoFrame]:
        if self.capture is None:
            raise VideoReaderError("VideoReader is not open")
        while True:
            started = perf_counter()
            ok, image = self.capture.read()
            decode_ms = (perf_counter() - started) * 1000
            if not ok:
                if self.next_index < self.frame_count - 1:
                    raise VideoReaderError(f"Video decode failed at frame {self.next_index} of {self.frame_count}")
                break
            index = self.next_index
            self.next_index += 1
            # OpenCV exposes the decoder's current presentation time in
            # milliseconds for containers/backends that preserve it. Fall back
            # to the nominal frame clock only when no useful PTS is available.
            position_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            if np.isfinite(position_ms) and position_ms >= 0 and (position_ms > 0 or index == 0):
                pts = int(round(position_ms))
                timestamp_seconds = position_ms / 1000.0
                time_base = "1/1000"
            else:
                pts = index
                timestamp_seconds = index / self.fps
                time_base = f"1/{self.fps:.12g}"
            yield VideoFrame(index=index, timestamp_seconds=timestamp_seconds, image=image, decode_ms=decode_ms, pts=pts, time_base=time_base)

    def seek(self, frame_index: int) -> None:
        if self.capture is None:
            raise VideoReaderError("VideoReader is not open")
        if frame_index < 0 or frame_index >= self.frame_count:
            raise VideoReaderError(f"Frame index is outside the video: {frame_index}")
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise VideoReaderError(f"Decoder could not seek to frame {frame_index}")
        self.next_index = frame_index

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, *_args) -> None:
        self.close()
