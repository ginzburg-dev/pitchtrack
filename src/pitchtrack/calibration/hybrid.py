from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchtrack.calibration.flow import (
    OpticalFlowConfig,
    estimate_interframe_motion,
    track_static_features,
)
from pitchtrack.calibration.temporal import (
    TemporalCalibrationConfig,
    TemporalCalibrationManager,
)
from pitchtrack.field_observation.base import FieldObservation
from pitchtrack.schemas import (
    CalibrationDiagnostics,
    CalibrationResult,
    CalibrationStatus,
)


@dataclass(frozen=True, slots=True)
class HybridCalibrationConfig:
    temporal: TemporalCalibrationConfig
    flow: OpticalFlowConfig = OpticalFlowConfig()


class HybridTemporalCalibrator:
    def __init__(self, config: HybridCalibrationConfig) -> None:
        self.config = config
        self.temporal = TemporalCalibrationManager(config.temporal)
        self._previous_frame: np.ndarray | None = None
        self._previous_result = CalibrationResult.invalid("not_initialized")

    def update(
        self,
        frame_bgr: np.ndarray,
        moving_mask: np.ndarray | None,
        observation: FieldObservation | None,
        *,
        frame_index: int,
    ) -> CalibrationResult:
        propagated: CalibrationResult | None = None
        if self._previous_frame is not None and self._previous_result.H_img_to_pitch is not None:
            flow = track_static_features(
                self._previous_frame,
                frame_bgr,
                moving_mask,
                self.config.flow,
            )
            height, width = frame_bgr.shape[:2]
            motion = estimate_interframe_motion(
                flow,
                frame_size_px=(width, height),
                config=self.config.flow,
            )
            if motion.valid and motion.H_current_to_previous is not None:
                matrix = self._previous_result.H_img_to_pitch @ motion.H_current_to_previous
                matrix = matrix / matrix[2, 2]
                diagnostics = CalibrationDiagnostics(
                    inlier_ratio=motion.inlier_ratio,
                    reprojection_error=motion.reprojection_error_px,
                    reprojection_error_unit="px",
                    spatial_coverage=motion.spatial_coverage,
                    temporal_consistency=motion.inlier_ratio,
                    last_valid_age_frames=(
                        self._previous_result.diagnostics.last_valid_age_frames + 1
                    ),
                )
                propagated = CalibrationResult(
                    status=CalibrationStatus.PREDICTED,
                    H_img_to_pitch=matrix,
                    confidence=max(
                        0.0,
                        min(
                            1.0,
                            self._previous_result.confidence * motion.inlier_ratio,
                        ),
                    ),
                    diagnostics=diagnostics,
                    reason="optical_flow",
                )

        result = self.temporal.update(
            frame_bgr,
            moving_mask,
            observation,
            frame_index=frame_index,
            propagated=propagated,
        )
        self._previous_frame = frame_bgr.copy()
        self._previous_result = result
        return result
