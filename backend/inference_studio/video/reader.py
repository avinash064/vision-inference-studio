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


class VideoReader:
    """Small sequential OpenCV decoder boundary, replaceable by PyAV/FFmpeg."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.capture: cv2.VideoCapture | None = None
        self.fps = 0.0
        self.frame_count = 0
        self.width = 0
        self.height = 0
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
        self.next_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if self.fps <= 0 or self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            self.close()
            raise VideoReaderError("Video has invalid or empty stream metadata")
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
            yield VideoFrame(index=index, timestamp_seconds=index / self.fps, image=image, decode_ms=decode_ms)

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
