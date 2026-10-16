from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchtrack.calibration.geometry import (
    HomographyConfig,
    estimate_homography,
    project_points,
)
from pitchtrack.calibration.state import CalibrationStateConfig
from pitchtrack.field_observation.base import FieldObservation
from pitchtrack.schemas import (
    CalibrationDiagnostics,
    CalibrationResult,
    CalibrationStatus,
)


@dataclass(frozen=True, slots=True)
class TemporalCalibrationConfig:
    homography: HomographyConfig
    state: CalibrationStateConfig = CalibrationStateConfig()


class TemporalCalibrationManager:
    def __init__(self, config: TemporalCalibrationConfig) -> None:
        self.config = config
        self._last_valid: CalibrationResult | None = None
        self._last_valid_frame: int | None = None
        self._last_frame_index: int | None = None
        self._current = CalibrationResult.invalid("not_initialized")

    @property
    def current(self) -> CalibrationResult:
        return self._current

    def should_request_observation(self, frame_index: int) -> bool:
        if self._last_valid_frame is None:
            return True
        return (
            frame_index - self._last_valid_frame
            >= self.config.state.reinitialize_interval_frames
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        moving_mask: np.ndarray | None,
        observation: FieldObservation | None,
        *,
        frame_index: int,
        propagated: CalibrationResult | None = None,
    ) -> CalibrationResult:
        del frame_bgr, moving_mask
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase monotonically")
        self._last_frame_index = frame_index

        if observation is not None and not observation.is_empty:
            candidate = estimate_homography(
                observation.image_points_px,
                observation.pitch_points_m,
                self.config.homography,
            )
            if candidate.status is CalibrationStatus.VALID:
                reference = self._candidate_reference(propagated)
                if reference is not None and self._anchor_disagreement(candidate, reference) > self.config.state.max_anchor_disagreement:
                    self._current = self._predict(
                        frame_index,
                        propagated,
                        reason="semantic_disagreement",
                    )
                    return self._current
                self._last_valid = candidate
                self._last_valid_frame = frame_index
                self._current = candidate
                return candidate

        self._current = self._predict(frame_index, propagated, reason="observation_missing")
        return self._current

    def _usable_reference(
        self, propagated: CalibrationResult | None
    ) -> CalibrationResult | None:
        if propagated is not None and propagated.H_img_to_pitch is not None:
            return propagated
        return self._last_valid

    def _candidate_reference(
        self, propagated: CalibrationResult | None
    ) -> CalibrationResult | None:
        if propagated is not None and propagated.H_img_to_pitch is not None:
            return propagated
        if self._current.status is CalibrationStatus.INVALID:
            return None
        if self._current.H_img_to_pitch is not None:
            return self._current
        return None

    def _predict(
        self,
        frame_index: int,
        propagated: CalibrationResult | None,
        *,
        reason: str,
    ) -> CalibrationResult:
        if self._last_valid is None or self._last_valid_frame is None:
            return CalibrationResult.invalid("no_valid_calibration")
        age = frame_index - self._last_valid_frame
        if age > self.config.state.max_prediction_age_frames:
            return CalibrationResult.invalid("prediction_expired")
        source = self._usable_reference(propagated)
        assert source is not None and source.H_img_to_pitch is not None
        diagnostics = CalibrationDiagnostics(
            inlier_ratio=source.diagnostics.inlier_ratio,
            reprojection_error=source.diagnostics.reprojection_error,
            reprojection_error_unit=source.diagnostics.reprojection_error_unit,
            spatial_coverage=source.diagnostics.spatial_coverage,
            temporal_consistency=source.diagnostics.temporal_consistency,
            last_valid_age_frames=age,
        )
        confidence = self._last_valid.confidence * (
            self.config.state.prediction_confidence_decay**age
        )
        return CalibrationResult(
            status=CalibrationStatus.PREDICTED,
            H_img_to_pitch=source.H_img_to_pitch,
            confidence=confidence,
            diagnostics=diagnostics,
            reason=reason,
        )

    def _anchor_disagreement(
        self,
        candidate: CalibrationResult,
        reference: CalibrationResult,
    ) -> float:
        width, height = self.config.homography.image_size_px
        anchors = np.asarray(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
                [float(width - 1) / 2.0, float(height - 1) / 2.0],
            ],
            dtype=np.float64,
        )
        candidate_points = project_points(anchors, candidate.H_img_to_pitch)
        reference_points = project_points(anchors, reference.H_img_to_pitch)
        return float(np.mean(np.linalg.norm(candidate_points - reference_points, axis=1)))
