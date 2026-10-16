from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from pitchtrack.schemas import CalibrationStatus


@dataclass(frozen=True, slots=True)
class FixturePaths:
    video_path: Path
    config_path: Path


def create_demo_fixture(
    output_dir: str | Path,
    *,
    frame_count: int = 12,
    fps: float = 10.0,
    calibration_status: CalibrationStatus = CalibrationStatus.VALID,
) -> FixturePaths:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "input.mp4"
    width, height = 320, 180
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create the synthetic fixture video")
    boxes: list[list[list[float]]] = []
    for frame_index in range(frame_count):
        frame = np.full((height, width, 3), (45, 105, 45), dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (300, 160), (240, 240, 240), 2)
        cv2.line(frame, (160, 20), (160, 160), (240, 240, 240), 2)
        x1 = 55 + frame_index * 3
        box = [float(x1), 65.0, float(x1 + 18), 115.0]
        boxes.append([box])
        cv2.rectangle(frame, (x1, 65), (x1 + 18, 115), (180, 30, 30), -1)
        writer.write(frame)
    writer.release()

    config = {
        "fixture": True,
        "detector": {"boxes_by_frame": boxes},
        "calibration": {
            "status": calibration_status.value,
            "image_points_px": [[20, 20], [300, 20], [300, 160], [20, 160]],
            "pitch_points_m": [[0, 0], [105, 0], [105, 68], [0, 68]],
            "pitch_width_m": 105.0,
            "pitch_height_m": 68.0,
            "reason": "fixture_requested_invalid",
        },
    }
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return FixturePaths(video_path=video_path, config_path=config_path)

