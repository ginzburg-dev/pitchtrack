from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pitchtrack.schemas import Detection, TrackObservation


@dataclass(frozen=True, slots=True)
class ByteTrackConfig:
    max_age_frames: int = 30
    min_iou: float = 0.2

    def __post_init__(self) -> None:
        if self.max_age_frames < 0:
            raise ValueError("max_age_frames must be non-negative")
        if not 0.0 <= self.min_iou <= 1.0:
            raise ValueError("min_iou must be between 0 and 1")


@dataclass(slots=True)
class _Track:
    track_id: int
    detection: Detection
    age: int = 0


class ByteTrackAdapter:
    """Deterministic IoU association with ByteTrack-style lost-track retention."""

    def __init__(self, config: ByteTrackConfig = ByteTrackConfig()) -> None:
        self.config = config
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(
        self, detections: Sequence[Detection], frame_bgr: np.ndarray
    ) -> list[TrackObservation]:
        del frame_bgr
        for track in self._tracks.values():
            track.age += 1
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        candidates = sorted(
            (
                (_iou(track.detection, detections[index]), track_id, index)
                for track_id, track in self._tracks.items()
                for index in range(len(detections))
            ),
            reverse=True,
        )
        assigned: dict[int, int] = {}
        for score, track_id, index in candidates:
            if score < self.config.min_iou:
                break
            if track_id in unmatched_tracks and index in unmatched_detections:
                assigned[index] = track_id
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(index)
        for index in sorted(unmatched_detections):
            assigned[index] = self._next_id
            self._tracks[self._next_id] = _Track(self._next_id, detections[index])
            self._next_id += 1
        output = []
        for index, detection in enumerate(detections):
            track_id = assigned[index]
            track = self._tracks[track_id]
            track.detection = detection
            track.age = 0
            output.append(TrackObservation(track_id, detection))
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.age <= self.config.max_age_frames
        }
        return output


def _iou(left: Detection, right: Detection) -> float:
    ax1, ay1, ax2, ay2 = left.bbox_xyxy_px
    bx1, by1, bx2, by2 = right.bbox_xyxy_px
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    left_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    right_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
