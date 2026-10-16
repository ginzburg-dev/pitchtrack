from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pitchtrack.demo.selector import PlayerBox
from pitchtrack.demo.manifest import DemoAssets, DemoManifest
from pitchtrack.demo.selector import (
    ClipSelectionCriteria,
    FrameCandidate,
    select_demo_interval,
)


@dataclass(frozen=True, slots=True)
class CropWindow:
    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


@dataclass(frozen=True, slots=True)
class DemoGenerationResult:
    video_path: Path
    transforms_path: Path
    quality_report_path: Path
    attribution_path: Path


def _window_from_center(
    center_x: float,
    center_y: float,
    width: float,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> CropWindow:
    source_width, source_height = source_size
    aspect = output_size[0] / output_size[1]
    width = min(float(source_width), max(2.0, width))
    height = width / aspect
    if height > source_height:
        height = float(source_height)
        width = height * aspect
    x = min(max(0.0, center_x - width / 2.0), source_width - width)
    y = min(max(0.0, center_y - height / 2.0), source_height - height)
    return CropWindow(x, y, width, height)


def compute_target_window(
    boxes: tuple[PlayerBox, ...],
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int] = (1920, 1080),
    margin_fraction: float = 0.2,
) -> CropWindow:
    if not boxes:
        raise ValueError("at least one player box is required for virtual PTZ framing")
    x1 = min(box.x1 for box in boxes)
    y1 = min(box.y1 for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    content_width = max(2.0, x2 - x1) * (1.0 + 2.0 * margin_fraction)
    content_height = max(2.0, y2 - y1) * (1.0 + 2.0 * margin_fraction)
    aspect = output_size[0] / output_size[1]
    width = max(content_width, content_height * aspect)
    return _window_from_center(
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        width,
        source_size=source_size,
        output_size=output_size,
    )


def smooth_crop_windows(
    targets: list[CropWindow],
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int] = (1920, 1080),
    max_pan_px_per_frame: float = 40.0,
    max_zoom_fraction_per_frame: float = 0.05,
    smoothing_alpha: float = 0.35,
) -> list[CropWindow]:
    if not targets:
        return []
    smoothed = [
        _window_from_center(
            targets[0].center_x,
            targets[0].center_y,
            targets[0].width,
            source_size=source_size,
            output_size=output_size,
        )
    ]
    for target in targets[1:]:
        previous = smoothed[-1]
        dx = (target.center_x - previous.center_x) * smoothing_alpha
        dy = (target.center_y - previous.center_y) * smoothing_alpha
        distance = math.hypot(dx, dy)
        if distance > max_pan_px_per_frame:
            scale = max_pan_px_per_frame / distance
            dx *= scale
            dy *= scale
        desired_width = previous.width + (target.width - previous.width) * smoothing_alpha
        max_delta = previous.width * max_zoom_fraction_per_frame
        width = min(max(desired_width, previous.width - max_delta), previous.width + max_delta)
        smoothed.append(
            _window_from_center(
                previous.center_x + dx,
                previous.center_y + dy,
                width,
                source_size=source_size,
                output_size=output_size,
            )
        )
    return smoothed


def crop_to_source_homography(
    window: CropWindow,
    output_size: tuple[int, int],
) -> np.ndarray:
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output_size must be positive")
    return np.asarray(
        [
            [window.width / output_width, 0.0, window.x],
            [0.0, window.height / output_height, window.y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compose_crop_to_pitch(
    H_source_to_pitch: np.ndarray,
    H_crop_to_source: np.ndarray,
) -> np.ndarray:
    result = np.asarray(H_source_to_pitch, dtype=np.float64) @ np.asarray(
        H_crop_to_source, dtype=np.float64
    )
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("homographies must compose to a finite 3x3 matrix")
    return result / result[2, 2]


def render_virtual_ptz_video(
    source_path: str | Path,
    output_path: str | Path,
    transforms_path: str | Path,
    windows: list[CropWindow],
    *,
    output_size: tuple[int, int] = (1920, 1080),
    H_source_to_pitch: np.ndarray | None = None,
    start_frame: int = 0,
) -> None:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create virtual PTZ output video")
    records: list[dict[str, Any]] = []
    try:
        for output_index, window in enumerate(windows):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Source ended before virtual PTZ frame {output_index}"
                )
            x1 = int(round(window.x))
            y1 = int(round(window.y))
            x2 = int(round(window.x + window.width))
            y2 = int(round(window.y + window.height))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise RuntimeError(f"Empty crop at output frame {output_index}")
            writer.write(cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR))
            crop_to_source = crop_to_source_homography(window, output_size)
            crop_to_pitch = (
                compose_crop_to_pitch(H_source_to_pitch, crop_to_source)
                if H_source_to_pitch is not None
                else None
            )
            records.append(
                {
                    "frame_index": output_index,
                    "source_frame_index": start_frame + output_index,
                    "crop_window_source_px": [
                        window.x,
                        window.y,
                        window.width,
                        window.height,
                    ],
                    "H_crop_to_source": crop_to_source.tolist(),
                    "H_crop_to_pitch": (
                        crop_to_pitch.tolist() if crop_to_pitch is not None else None
                    ),
                }
            )
    finally:
        writer.release()
        capture.release()
    Path(transforms_path).write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _load_frame_candidates(path: Path) -> list[FrameCandidate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        frames = data.get("frames", data.get("annotations", []))
    else:
        frames = data
    candidates: list[FrameCandidate] = []
    for fallback_index, frame in enumerate(frames):
        entities = frame.get("players", frame.get("entities", []))
        boxes: list[PlayerBox] = []
        for entity in entities:
            bbox = entity.get(
                "bbox_xyxy_px",
                entity.get("bbox_image", entity.get("bbox")),
            )
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if x2 > x1 and y2 > y1:
                boxes.append(PlayerBox(x1, y1, x2, y2))
        candidates.append(
            FrameCandidate(
                frame_index=int(
                    frame.get("frame_index", frame.get("image_id", fallback_index))
                ),
                player_boxes=tuple(boxes),
                line_visible=bool(frame.get("line_visible", True)),
                lighting_score=float(frame.get("lighting_score", 1.0)),
            )
        )
    return candidates


def generate_demo_clip(
    assets: DemoAssets,
    manifest: DemoManifest,
    output_dir: str | Path,
) -> DemoGenerationResult:
    capture = cv2.VideoCapture(str(assets.video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open source video: {assets.video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    candidates = _load_frame_candidates(assets.annotations_path)
    gates = manifest.visual_gates
    criteria = ClipSelectionCriteria(
        fps=fps,
        duration_s=manifest.duration_s,
        min_players=int(gates["min_players"]),
        min_player_height_px=gates["min_player_height_px"],
        max_player_height_px=gates["max_player_height_px"],
        min_passing_fraction=gates["min_passing_fraction"],
        min_lighting_score=gates["min_lighting_score"],
    )
    selection = select_demo_interval(candidates, criteria)
    selected = [
        candidate
        for candidate in candidates
        if selection.start_frame
        <= candidate.frame_index
        < selection.end_frame_exclusive
    ]
    targets = [
        compute_target_window(
            candidate.player_boxes,
            source_size=source_size,
            output_size=manifest.output_size,
        )
        for candidate in selected
    ]
    windows = smooth_crop_windows(
        targets,
        source_size=source_size,
        output_size=manifest.output_size,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "pitchtrack-demo-virtual-ptz.mp4"
    transforms_path = output / "pitchtrack-demo-virtual-ptz.transforms.jsonl"
    quality_path = output / "clip-quality.json"
    attribution_path = output / "attribution.json"
    render_virtual_ptz_video(
        assets.video_path,
        video_path,
        transforms_path,
        windows,
        output_size=manifest.output_size,
        start_frame=selection.start_frame,
    )
    quality_path.write_text(
        json.dumps(
            {
                "start_frame": selection.start_frame,
                "end_frame_exclusive": selection.end_frame_exclusive,
                "selected_frame_count": len(selected),
                "passing_fraction": selection.passing_fraction,
                "median_player_height_px": selection.median_player_height_px,
                "output_size": list(manifest.output_size),
                "virtual_camera": True,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    attribution_path.write_text(
        json.dumps(manifest.attribution(), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return DemoGenerationResult(
        video_path,
        transforms_path,
        quality_path,
        attribution_path,
    )

