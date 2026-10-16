from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pitchtrack.schemas import CalibrationResult, Detection, TrackObservation


class ScriptedDetector:
    def __init__(self, boxes_by_frame: Sequence[Sequence[Sequence[float]]]) -> None:
        self._boxes = list(boxes_by_frame)
        self._frame_index = 0

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        del frame_bgr
        boxes = self._boxes[min(self._frame_index, len(self._boxes) - 1)]
        self._frame_index += 1
        return [Detection(tuple(box), 0.95) for box in boxes]


class StableSlotTracker:
    def update(
        self,
        detections: Sequence[Detection],
        frame_bgr: np.ndarray,
    ) -> list[TrackObservation]:
        del frame_bgr
        return [
            TrackObservation(track_id=index + 1, detection=detection)
            for index, detection in enumerate(detections)
        ]


class StaticCalibrator:
    def __init__(self, result: CalibrationResult) -> None:
        self.result = result

    def update(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        detections: Sequence[Detection] = (),
    ) -> CalibrationResult:
        del frame_bgr, frame_index, detections
        return self.result


class HybridFixtureCalibrationSource:
    def __init__(self, calibrator: Any, first_observation: Any) -> None:
        self.calibrator = calibrator
        self.first_observation = first_observation

    def update(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        detections: Sequence[Detection] = (),
    ) -> CalibrationResult:
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        for detection in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy_px)
            x1 = max(0, min(mask.shape[1] - 1, x1))
            x2 = max(0, min(mask.shape[1], x2 + 1))
            y1 = max(0, min(mask.shape[0] - 1, y1))
            y2 = max(0, min(mask.shape[0], y2 + 1))
            mask[y1:y2, x1:x2] = 255
        observation = self.first_observation if frame_index == 0 else None
        return self.calibrator.update(
            frame_bgr,
            mask,
            observation,
            frame_index=frame_index,
        )
