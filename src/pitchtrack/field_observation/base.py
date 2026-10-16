from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


def _point_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass(frozen=True, slots=True)
class FieldObservation:
    image_points_px: np.ndarray
    pitch_points_m: np.ndarray
    confidences: np.ndarray
    source: str

    def __post_init__(self) -> None:
        image = _point_array(self.image_points_px, "image_points_px")
        pitch = _point_array(self.pitch_points_m, "pitch_points_m")
        confidences = np.asarray(self.confidences, dtype=np.float64)
        if len(image) != len(pitch) or len(image) != len(confidences):
            raise ValueError(
                "image points, pitch points, and confidences must have the same number of entries"
            )
        if confidences.ndim != 1 or not np.all(np.isfinite(confidences)):
            raise ValueError("confidences must be a finite one-dimensional array")
        if np.any((confidences < 0.0) | (confidences > 1.0)):
            raise ValueError("confidences must be between 0 and 1")
        object.__setattr__(self, "image_points_px", image)
        object.__setattr__(self, "pitch_points_m", pitch)
        object.__setattr__(self, "confidences", confidences.copy())

    @property
    def is_empty(self) -> bool:
        return len(self.image_points_px) == 0

    @classmethod
    def empty(cls, source: str) -> "FieldObservation":
        return cls(
            image_points_px=np.empty((0, 2), dtype=np.float64),
            pitch_points_m=np.empty((0, 2), dtype=np.float64),
            confidences=np.empty((0,), dtype=np.float64),
            source=source,
        )


@runtime_checkable
class FieldObservationProvider(Protocol):
    def observe(self, frame_bgr: np.ndarray) -> FieldObservation: ...
