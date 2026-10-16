from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoOpenError(RuntimeError):
    """Raised when OpenCV cannot open a video input."""


class VideoDecodeError(RuntimeError):
    """Raised when decoding ends before the container's declared frame count."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    declared_frame_count: int


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame_index: int
    timestamp_s: float
    image_bgr: np.ndarray


class VideoReader:
    def __init__(self, path: str | Path, *, strict_frame_count: bool = True) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video does not exist: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise VideoOpenError(f"Could not open video: {self.path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        width = int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        declared = int(round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            self._capture.release()
            raise VideoOpenError(f"Video has invalid stream metadata: {self.path}")
        self.metadata = VideoMetadata(fps, width, height, max(0, declared))
        self.strict_frame_count = strict_frame_count
        self._closed = False

    def __iter__(self) -> Iterator[VideoFrame]:
        decoded = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            if image is None or image.size == 0:
                self.close()
                raise VideoDecodeError(
                    f"Decoder returned an empty frame at index {decoded}: {self.path}"
                )
            yield VideoFrame(decoded, decoded / self.metadata.fps, image)
            decoded += 1
        if (
            self.strict_frame_count
            and self.metadata.declared_frame_count > 0
            and decoded < self.metadata.declared_frame_count
        ):
            raise VideoDecodeError(
                f"Video ended after {decoded} of "
                f"{self.metadata.declared_frame_count} declared frames: {self.path}"
            )

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

