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
    camera_pan_px_per_frame: float = 0.0,
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
    base = np.full((height, width, 3), (45, 105, 45), dtype=np.uint8)
    cv2.rectangle(base, (20, 20), (300, 160), (240, 240, 240), 2)
    cv2.line(base, (160, 20), (160, 160), (240, 240, 240), 2)
    for x in range(30, width - 20, 20):
        for y in range(30, height - 20, 20):
            cv2.drawMarker(base, (x, y), (180, 210, 180), cv2.MARKER_CROSS, 5, 1)

    boxes: list[list[list[float]]] = []
    ground_truth: list[list[list[float]]] = []
    initial_image_points = np.asarray(
        [[20.0, 20.0], [300.0, 20.0], [300.0, 160.0], [20.0, 160.0]],
        dtype=np.float64,
    )
    pitch_points = np.asarray(
        [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
        dtype=np.float64,
    )
    initial_h, _ = cv2.findHomography(initial_image_points, pitch_points, 0)
    for frame_index in range(frame_count):
        tx = camera_pan_px_per_frame * frame_index
        camera_transform = np.asarray(
            [[1.0, 0.0, tx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        frame = cv2.warpPerspective(base, camera_transform, (width, height))
        x1 = 55 + frame_index * 3
        box = [float(x1), 65.0, float(x1 + 18), 115.0]
        boxes.append([box])
        cv2.rectangle(frame, (x1, 65), (x1 + 18, 115), (180, 30, 30), -1)
        writer.write(frame)
        h_current = initial_h @ np.linalg.inv(camera_transform)
        ground_truth.append((h_current / h_current[2, 2]).tolist())
    writer.release()

    config = {
        "fixture": True,
        "detector": {"boxes_by_frame": boxes},
        "calibration": {
            "status": calibration_status.value,
            "image_points_px": initial_image_points.tolist(),
            "pitch_points_m": pitch_points.tolist(),
            "pitch_width_m": 105.0,
            "pitch_height_m": 68.0,
            "reason": "fixture_requested_invalid",
        },
        "video": {"width": width, "height": height, "fps": fps},
        "ground_truth_H_img_to_pitch": ground_truth,
    }
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return FixturePaths(video_path=video_path, config_path=config_path)

