import cv2
import numpy as np

from pitchtrack.calibration.flow import OpticalFlowConfig
from pitchtrack.calibration.geometry import HomographyConfig, project_points
from pitchtrack.calibration.hybrid import HybridCalibrationConfig, HybridTemporalCalibrator
from pitchtrack.calibration.state import CalibrationStateConfig
from pitchtrack.calibration.temporal import TemporalCalibrationConfig
from pitchtrack.field_observation.base import FieldObservation
from pitchtrack.schemas import CalibrationStatus


WIDTH, HEIGHT = 320, 240
IMAGE_POINTS = np.array(
    [[20.0, 20.0], [300.0, 20.0], [300.0, 220.0], [20.0, 220.0]],
    dtype=np.float64,
)
PITCH_POINTS = np.array(
    [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
    dtype=np.float64,
)


def field_frame() -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (42, 110, 42), dtype=np.uint8)
    for x in range(20, WIDTH - 19, 20):
        for y in range(20, HEIGHT - 19, 20):
            cv2.drawMarker(frame, (x, y), (235, 235, 235), cv2.MARKER_CROSS, 7, 1)
    cv2.rectangle(frame, (20, 20), (300, 220), (255, 255, 255), 2)
    return frame


def observation(image_points: np.ndarray = IMAGE_POINTS) -> FieldObservation:
    return FieldObservation(
        image_points_px=image_points,
        pitch_points_m=PITCH_POINTS,
        confidences=np.ones(4),
        source="synthetic-lines",
    )


def calibrator(max_age: int = 4) -> HybridTemporalCalibrator:
    return HybridTemporalCalibrator(
        HybridCalibrationConfig(
            temporal=TemporalCalibrationConfig(
                homography=HomographyConfig(
                    image_size_px=(WIDTH, HEIGHT),
                    min_spatial_coverage=0.1,
                    max_reprojection_error=0.01,
                ),
                state=CalibrationStateConfig(
                    max_prediction_age_frames=max_age,
                    prediction_confidence_decay=0.8,
                    reinitialize_interval_frames=3,
                    max_anchor_disagreement=8.0,
                ),
            ),
            flow=OpticalFlowConfig(
                max_corners=400,
                min_points=12,
                min_spatial_coverage=0.08,
            ),
        )
    )


def test_hybrid_calibration_propagates_pan_and_zoom_by_projected_anchors() -> None:
    first = field_frame()
    transform = np.array(
        [[1.02, 0.0, 4.0], [0.0, 1.02, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    second = cv2.warpPerspective(first, transform, (WIDTH, HEIGHT))
    hybrid = calibrator()

    valid = hybrid.update(first, None, observation(), frame_index=0)
    predicted = hybrid.update(second, None, None, frame_index=1)

    anchors = np.array([[40.0, 40.0], [160.0, 120.0], [270.0, 190.0]])
    expected_h = valid.H_img_to_pitch @ np.linalg.inv(transform)
    expected = project_points(anchors, expected_h)
    actual = project_points(anchors, predicted.H_img_to_pitch)

    assert predicted.status is CalibrationStatus.PREDICTED
    assert np.mean(np.linalg.norm(actual - expected, axis=1)) < 0.6


def test_hybrid_calibration_expires_during_dropout_and_recovers_semantically() -> None:
    first = field_frame()
    hybrid = calibrator(max_age=1)
    hybrid.update(first, None, observation(), frame_index=0)
    predicted = hybrid.update(first, None, None, frame_index=1)
    invalid = hybrid.update(first, None, None, frame_index=2)

    shifted_points = IMAGE_POINTS + np.array([12.0, 0.0])
    shifted = cv2.warpAffine(
        first,
        np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (WIDTH, HEIGHT),
    )
    recovered = hybrid.update(shifted, None, observation(shifted_points), frame_index=3)

    assert predicted.status is CalibrationStatus.PREDICTED
    assert invalid.status is CalibrationStatus.INVALID
    assert recovered.status is CalibrationStatus.VALID

