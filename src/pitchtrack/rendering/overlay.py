from __future__ import annotations

import cv2
import numpy as np

from pitchtrack.rendering.pitch import draw_pitch
from pitchtrack.schemas import CalibrationResult, CalibrationStatus, TrackObservation


def render_annotated_frame(
    image_bgr: np.ndarray,
    tracks: tuple[TrackObservation, ...],
    calibration: CalibrationResult,
) -> np.ndarray:
    annotated = image_bgr.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.detection.bbox_xyxy_px)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
        label = f"ID {track.track_id}" if track.track_id is not None else "ID ?"
        cv2.putText(
            annotated,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    status_color = {
        CalibrationStatus.VALID: (60, 220, 60),
        CalibrationStatus.PREDICTED: (0, 200, 255),
        CalibrationStatus.INVALID: (40, 40, 230),
    }[calibration.status]
    status_text = f"CALIBRATION: {calibration.status.value.upper()} {calibration.confidence:.2f}"
    cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 360), 34), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        status_text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )
    panel_width = max(240, int(round(annotated.shape[0] * 1.5)))
    pitch = draw_pitch(tracks, width=panel_width, height=annotated.shape[0])
    return np.concatenate([annotated, pitch], axis=1)
