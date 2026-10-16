import json

import numpy as np

from pitchtrack.schemas import (
    CalibrationDiagnostics,
    CalibrationResult,
    CalibrationStatus,
    Detection,
    FrameResult,
    TrackObservation,
)


def test_detection_footpoint_is_bbox_bottom_center() -> None:
    detection = Detection(
        bbox_xyxy_px=(10.0, 20.0, 30.0, 80.0),
        confidence=0.75,
        class_name="person",
    )

    assert detection.foot_xy_px == (20.0, 80.0)


def test_invalid_calibration_serializes_track_pitch_coordinates_as_null() -> None:
    calibration = CalibrationResult.invalid("not_enough_correspondences")
    track = TrackObservation(
        track_id=7,
        detection=Detection((10.0, 20.0, 30.0, 80.0), 0.75),
        field_xy_m=None,
    )
    frame = FrameResult(
        frame_index=3,
        timestamp_s=0.12,
        tracks=(track,),
        calibration=calibration,
    )

    payload = frame.to_dict()
    encoded = json.dumps(payload, allow_nan=False)

    assert payload["calibration"]["status"] == "invalid"
    assert payload["tracks"][0]["field_xy_m"] is None
    assert '"field_xy_m": null' in encoded


def test_calibration_matrix_serializes_to_plain_nested_lists() -> None:
    calibration = CalibrationResult(
        status=CalibrationStatus.VALID,
        H_img_to_pitch=np.eye(3, dtype=np.float64),
        confidence=0.9,
        diagnostics=CalibrationDiagnostics(
            inlier_ratio=1.0,
            reprojection_error=0.0,
            reprojection_error_unit="m",
            spatial_coverage=0.5,
            temporal_consistency=1.0,
            last_valid_age_frames=0,
        ),
    )

    payload = calibration.to_dict()

    assert payload["H_img_to_pitch"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    json.dumps(payload, allow_nan=False)


def test_non_finite_schema_values_are_rejected() -> None:
    try:
        Detection((0.0, 0.0, float("nan"), 1.0), 0.5)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite bounding box must be rejected")

