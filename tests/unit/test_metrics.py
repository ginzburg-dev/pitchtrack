import numpy as np
import pytest

from pitchtrack.evaluation.calibration import (
    ReprojectionSample,
    completeness,
    mean_reprojection_error,
    tolerant_jaccard,
)
from pitchtrack.evaluation.performance import dropped_frames, summarize_stage_latencies
from pitchtrack.evaluation.temporal import (
    calibration_recovery,
    response_lag_frames,
    status_coverage,
    trajectory_jitter,
)
from pitchtrack.evaluation.tracking import to_hota_inputs
from pitchtrack.projection.smoothing import PitchTrackSmoother
from pitchtrack.schemas import CalibrationStatus


def test_tolerant_jaccard_and_completeness_have_explicit_empty_semantics() -> None:
    target = np.zeros((20, 20), dtype=np.uint8)
    predicted = np.zeros_like(target)
    target[10, 5:15] = 1
    predicted[11, 5:15] = 1

    assert tolerant_jaccard(predicted, target, tolerance_px=1) == pytest.approx(1.0)
    assert tolerant_jaccard(np.zeros_like(target), np.zeros_like(target)) is None
    assert completeness([True, False, True]) == pytest.approx(2 / 3)
    assert completeness([]) is None


def test_reprojection_error_keeps_units_and_rejects_mixed_units() -> None:
    metric = mean_reprojection_error([ReprojectionSample(1.0, "m"), ReprojectionSample(3.0, "m")])
    assert metric == {"value": 2.0, "unit": "m", "sample_count": 2}
    assert mean_reprojection_error([]) == {
        "value": None,
        "unit": None,
        "sample_count": 0,
    }
    with pytest.raises(ValueError, match="mixed reprojection units"):
        mean_reprojection_error([ReprojectionSample(1.0, "px"), ReprojectionSample(1.0, "m")])


def test_status_coverage_and_recovery_time() -> None:
    statuses = [
        CalibrationStatus.VALID,
        CalibrationStatus.INVALID,
        CalibrationStatus.PREDICTED,
        CalibrationStatus.VALID,
        CalibrationStatus.INVALID,
    ]
    coverage = status_coverage(statuses)
    recovery = calibration_recovery(statuses)

    assert coverage["valid"] == pytest.approx(0.4)
    assert coverage["invalid"] == pytest.approx(0.4)
    assert recovery["recovery_frames"] == [2]
    assert recovery["unresolved_failures"] == 1


def test_smoothing_reduces_jitter_reports_lag_and_resets_on_invalid() -> None:
    smoother = PitchTrackSmoother(alpha=0.5)
    raw = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
    smoothed = [smoother.update(7, point, CalibrationStatus.VALID) for point in raw]

    assert all(point is not None for point in smoothed)
    assert trajectory_jitter(smoothed) < trajectory_jitter(raw)
    assert response_lag_frames(raw, smoothed, movement_threshold=1.5) >= 0
    assert smoother.update(7, (4.0, 1.0), CalibrationStatus.INVALID) is None
    assert smoother.update(7, (8.0, 2.0), CalibrationStatus.VALID) == (8.0, 2.0)


def test_hota_adapter_preserves_frame_ids_and_drops_missing_track_ids() -> None:
    frames = [
        [
            {"track_id": 3, "bbox_xyxy_px": [1, 2, 5, 8]},
            {"track_id": None, "bbox_xyxy_px": [8, 2, 9, 4]},
        ],
        [{"track_id": 3, "bbox_xyxy_px": [2, 2, 6, 8]}],
    ]

    result = to_hota_inputs(frames)

    assert result["ids"] == [[3], [3]]
    assert result["detections"][0] == [[1.0, 2.0, 4.0, 6.0]]


def test_latency_and_dropped_frames_do_not_fabricate_unavailable_values() -> None:
    metrics = summarize_stage_latencies({"detect": [0.010, 0.020, 0.030], "track": []})

    assert metrics["detect"]["p50_ms"] == pytest.approx(20.0)
    assert metrics["track"]["p50_ms"] is None
    assert dropped_frames(expected_frames=None, processed_frames=10) is None
    assert dropped_frames(expected_frames=12, processed_frames=10) == 2
