from __future__ import annotations

from importlib import import_module
from pathlib import Path

import cv2
import numpy as np


def video_metadata(path: str | Path) -> tuple[float, int, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open media source: {path}")
    metadata = (
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    capture.release()
    return metadata


def encode_gif(
    source: str | Path,
    destination: str | Path,
    *,
    start: int,
    stop: int,
    label: str,
    target_fps: float = 8.0,
    max_width: int = 960,
) -> None:
    try:
        image_module = import_module("PIL.Image")
    except ImportError as error:
        raise RuntimeError("GIF encoding requires the 'media' extra (Pillow).") from error
    fps, _, (width, height) = video_metadata(source)
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    step = max(1, int(round(fps / target_fps)))
    output_width = min(width, max_width)
    output_height = max(1, int(round(height * output_width / width)))
    frames = []
    for frame_index in range(start, stop):
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start) % step:
            continue
        frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        _label_frame(frame, label)
        frames.append(image_module.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise ValueError("selected GIF interval contains no decodable frames")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(round(1000 * step / fps))),
        loop=0,
        optimize=True,
        disposal=2,
    )


def encode_mp4(
    source: str | Path,
    destination: str | Path,
    *,
    start: int,
    stop: int,
    label: str,
) -> None:
    fps, _, size = video_metadata(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not create MP4 asset")
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(start, stop):
        ok, frame = capture.read()
        if not ok:
            break
        _label_frame(frame, label)
        writer.write(frame)
    capture.release()
    writer.release()


def enforce_asset_size(path: str | Path, *, max_bytes: int) -> None:
    size = Path(path).stat().st_size
    if size > max_bytes:
        raise ValueError(f"{path} exceeds the asset size limit ({size} > {max_bytes})")


def _label_frame(frame: np.ndarray, label: str) -> None:
    if not label:
        return
    height, width = frame.shape[:2]
    scale = max(0.4, min(width, height) / 900.0)
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    cv2.rectangle(frame, (8, 8), (20 + text_width, 22 + text_height), (255, 255, 255), -1)
    cv2.putText(
        frame,
        label,
        (14, 16 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (36, 41, 47),
        thickness,
        cv2.LINE_AA,
    )
