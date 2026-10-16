from __future__ import annotations

from collections.abc import Sequence

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

    def update(self, frame_bgr: np.ndarray, frame_index: int) -> CalibrationResult:
        del frame_bgr, frame_index
        return self.result

