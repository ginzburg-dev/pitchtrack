from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


def _finite_tuple(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not np.all(np.isfinite(np.asarray(result, dtype=np.float64))):
        raise ValueError(f"{name} must contain only finite values")
    return result


class CalibrationStatus(str, Enum):
    VALID = "valid"
    PREDICTED = "predicted"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy_px: tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"

    def __post_init__(self) -> None:
        bbox = _finite_tuple(self.bbox_xyxy_px, "bbox_xyxy_px")
        confidence = float(self.confidence)
        if not np.isfinite(confidence):
            raise ValueError("confidence must be finite")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("bbox_xyxy_px must use x1,y1,x2,y2 ordering")
        object.__setattr__(self, "bbox_xyxy_px", bbox)
        object.__setattr__(self, "confidence", confidence)

    @property
    def foot_xy_px(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox_xyxy_px
        return ((x1 + x2) / 2.0, y2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy_px": list(self.bbox_xyxy_px),
            "foot_xy_px": list(self.foot_xy_px),
        }


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    reprojection_error_unit: str = "unknown"
    spatial_coverage: float = 0.0
    temporal_consistency: float = 0.0
    last_valid_age_frames: int = 0

    def __post_init__(self) -> None:
        numeric = (
            self.inlier_ratio,
            self.reprojection_error,
            self.spatial_coverage,
            self.temporal_consistency,
        )
        if not np.all(np.isfinite(np.asarray(numeric, dtype=np.float64))):
            raise ValueError("calibration diagnostics must be finite")
        if self.last_valid_age_frames < 0:
            raise ValueError("last_valid_age_frames must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "inlier_ratio": float(self.inlier_ratio),
            "reprojection_error": float(self.reprojection_error),
            "reprojection_error_unit": self.reprojection_error_unit,
            "spatial_coverage": float(self.spatial_coverage),
            "temporal_consistency": float(self.temporal_consistency),
            "last_valid_age_frames": int(self.last_valid_age_frames),
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    status: CalibrationStatus
    H_img_to_pitch: np.ndarray | None
    confidence: float
    diagnostics: CalibrationDiagnostics = field(default_factory=CalibrationDiagnostics)
    reason: str | None = None

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("calibration confidence must be finite and between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        if self.H_img_to_pitch is None:
            if self.status is not CalibrationStatus.INVALID:
                raise ValueError("valid or predicted calibration requires H_img_to_pitch")
            return
        matrix = np.asarray(self.H_img_to_pitch, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("H_img_to_pitch must be a finite 3x3 matrix")
        if self.status is CalibrationStatus.INVALID:
            raise ValueError("invalid calibration must not carry a homography")
        object.__setattr__(self, "H_img_to_pitch", matrix.copy())

    @classmethod
    def invalid(
        cls,
        reason: str,
        diagnostics: CalibrationDiagnostics | None = None,
    ) -> "CalibrationResult":
        return cls(
            status=CalibrationStatus.INVALID,
            H_img_to_pitch=None,
            confidence=0.0,
            diagnostics=diagnostics or CalibrationDiagnostics(),
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "H_img_to_pitch": (
                self.H_img_to_pitch.tolist() if self.H_img_to_pitch is not None else None
            ),
            "confidence": self.confidence,
            "diagnostics": self.diagnostics.to_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: int | None
    detection: Detection
    field_xy_m: tuple[float, float] | None = None
    smoothed_field_xy_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.field_xy_m is not None:
            object.__setattr__(self, "field_xy_m", _finite_tuple(self.field_xy_m, "field_xy_m"))
        if self.smoothed_field_xy_m is not None:
            object.__setattr__(
                self,
                "smoothed_field_xy_m",
                _finite_tuple(self.smoothed_field_xy_m, "smoothed_field_xy_m"),
            )

    @property
    def foot_xy_px(self) -> tuple[float, float]:
        return self.detection.foot_xy_px

    def to_dict(self) -> dict[str, Any]:
        payload = self.detection.to_dict()
        payload.update(
            {
                "track_id": self.track_id,
                "field_xy_m": list(self.field_xy_m) if self.field_xy_m is not None else None,
                "smoothed_field_xy_m": (
                    list(self.smoothed_field_xy_m) if self.smoothed_field_xy_m is not None else None
                ),
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    timestamp_s: float
    tracks: tuple[TrackObservation, ...]
    calibration: CalibrationResult

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": float(self.timestamp_s),
            "tracks": [track.to_dict() for track in self.tracks],
            "calibration": self.calibration.to_dict(),
        }
