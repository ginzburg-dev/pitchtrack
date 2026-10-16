from __future__ import annotations

import numpy as np

from pitchtrack.calibration.geometry import project_points
from pitchtrack.schemas import CalibrationResult, CalibrationStatus, TrackObservation


class PitchProjector:
    def project(
        self,
        track: TrackObservation,
        calibration: CalibrationResult,
    ) -> TrackObservation:
        field_xy_m: tuple[float, float] | None = None
        if (
            calibration.status in {CalibrationStatus.VALID, CalibrationStatus.PREDICTED}
            and calibration.H_img_to_pitch is not None
        ):
            projected = project_points(
                np.asarray([track.foot_xy_px], dtype=np.float64),
                calibration.H_img_to_pitch,
            )[0]
            field_xy_m = (float(projected[0]), float(projected[1]))
        return TrackObservation(
            track_id=track.track_id,
            detection=track.detection,
            field_xy_m=field_xy_m,
            smoothed_field_xy_m=track.smoothed_field_xy_m,
        )

