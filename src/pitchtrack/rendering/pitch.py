from __future__ import annotations

import cv2
import numpy as np

from pitchtrack.schemas import TrackObservation


def draw_pitch(
    tracks: tuple[TrackObservation, ...],
    *,
    width: int,
    height: int,
    pitch_width_m: float = 105.0,
    pitch_height_m: float = 68.0,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (45, 105, 45), dtype=np.uint8)
    margin = max(8, min(width, height) // 15)
    left, top = margin, margin
    right, bottom = width - margin, height - margin
    cv2.rectangle(canvas, (left, top), (right, bottom), (240, 240, 240), 2)
    center_x = (left + right) // 2
    cv2.line(canvas, (center_x, top), (center_x, bottom), (240, 240, 240), 2)
    radius = max(5, int((bottom - top) * 9.15 / pitch_height_m))
    cv2.circle(canvas, (center_x, (top + bottom) // 2), radius, (240, 240, 240), 2)
    for track in tracks:
        if track.field_xy_m is None:
            continue
        x_m, y_m = track.field_xy_m
        x = int(round(left + (x_m / pitch_width_m) * (right - left)))
        y = int(round(top + (y_m / pitch_height_m) * (bottom - top)))
        color = (
            (20, 210, 255)
            if 0 <= x_m <= pitch_width_m and 0 <= y_m <= pitch_height_m
            else (0, 80, 255)
        )
        cv2.circle(canvas, (x, y), 5, color, -1)
        if track.track_id is not None:
            cv2.putText(
                canvas,
                str(track.track_id),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas
