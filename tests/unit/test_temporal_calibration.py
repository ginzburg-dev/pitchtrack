import numpy as np

from pitchtrack.calibration.geometry import HomographyConfig
from pitchtrack.calibration.state import CalibrationStateConfig
from pitchtrack.calibration.temporal import (
    TemporalCalibrationConfig,
    TemporalCalibrationManager,
)
from pitchtrack.field_observation.base import FieldObservation
from pitchtrack.projection.projector import PitchProjector
from pitchtrack.schemas import CalibrationStatus, Detection, TrackObservation


IMAGE_POINTS = np.array(
    [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
    dtype=np.float64,
)
PITCH_POINTS = np.array(
    [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
    dtype=np.float64,
)


def make_observation(pitch_offset_x: float = 0.0) -> FieldObservation:
    pitch = PITCH_POINTS.copy()
    pitch[:, 0] += pitch_offset_x
    return FieldObservation(
        image_points_px=IMAGE_POINTS,
        pitch_points_m=pitch,
        confidences=np.ones(4),
        source="synthetic",
    )


def make_manager(
    *,
    max_age: int = 2,
    decay: float = 0.5,
    interval: int = 3,
    max_disagreement: float = 5.0,
) -> TemporalCalibrationManager:
    return TemporalCalibrationManager(
        TemporalCalibrationConfig(
            homography=HomographyConfig(
                image_size_px=(100, 100),
                min_spatial_coverage=0.1,
                max_reprojection_error=0.01,
            ),
            state=CalibrationStateConfig(
                max_prediction_age_frames=max_age,
                prediction_confidence_decay=decay,
                reinitialize_interval_frames=interval,
                max_anchor_disagreement=max_disagreement,
            ),
        )
    )


def test_temporal_manager_transitions_valid_predicted_invalid() -> None:
    manager = make_manager(max_age=2, decay=0.5)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    valid = manager.update(frame, None, make_observation(), frame_index=0)
    predicted_1 = manager.update(frame, None, None, frame_index=1)
    predicted_2 = manager.update(frame, None, None, frame_index=2)
    invalid = manager.update(frame, None, None, frame_index=3)

    assert valid.status is CalibrationStatus.VALID
    assert predicted_1.status is CalibrationStatus.PREDICTED
    assert predicted_2.status is CalibrationStatus.PREDICTED
    assert invalid.status is CalibrationStatus.INVALID
    assert valid.confidence > predicted_1.confidence > predicted_2.confidence > 0.0
    assert invalid.confidence == 0.0
    assert [
        predicted_1.diagnostics.last_valid_age_frames,
        predicted_2.diagnostics.last_valid_age_frames,
    ] == [1, 2]


def test_invalid_after_prediction_age_produces_null_player_coordinates() -> None:
    manager = make_manager(max_age=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager.update(frame, None, make_observation(), frame_index=0)
    manager.update(frame, None, None, frame_index=1)
    invalid = manager.update(frame, None, None, frame_index=2)
    track = TrackObservation(1, Detection((10.0, 10.0, 20.0, 30.0), 0.9))

    projected = PitchProjector().project(track, invalid)

    assert invalid.status is CalibrationStatus.INVALID
    assert projected.field_xy_m is None


def test_semantic_candidate_with_large_anchor_disagreement_is_rejected() -> None:
    manager = make_manager(max_disagreement=5.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager.update(frame, None, make_observation(), frame_index=0)

    result = manager.update(
        frame,
        None,
        make_observation(pitch_offset_x=50.0),
        frame_index=1,
    )

    assert result.status is CalibrationStatus.PREDICTED
    assert result.reason == "semantic_disagreement"
    assert result.diagnostics.last_valid_age_frames == 1


def test_periodic_reinitialization_request_uses_last_valid_frame() -> None:
    manager = make_manager(interval=3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert manager.should_request_observation(frame_index=0)
    manager.update(frame, None, make_observation(), frame_index=0)
    assert not manager.should_request_observation(frame_index=1)
    assert not manager.should_request_observation(frame_index=2)
    assert manager.should_request_observation(frame_index=3)


def test_consistent_reinitialization_resets_prediction_age() -> None:
    manager = make_manager()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager.update(frame, None, make_observation(), frame_index=0)
    manager.update(frame, None, None, frame_index=1)

    result = manager.update(frame, None, make_observation(), frame_index=2)

    assert result.status is CalibrationStatus.VALID
    assert result.diagnostics.last_valid_age_frames == 0
