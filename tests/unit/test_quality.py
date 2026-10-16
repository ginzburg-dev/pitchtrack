from pitchtrack.calibration.quality import QualityConfig, score_quality
from pitchtrack.schemas import CalibrationDiagnostics


def diagnostics(**overrides: float | int | str) -> CalibrationDiagnostics:
    values: dict[str, float | int | str] = {
        "inlier_ratio": 0.9,
        "reprojection_error": 1.0,
        "reprojection_error_unit": "px",
        "spatial_coverage": 0.6,
        "temporal_consistency": 0.9,
        "last_valid_age_frames": 0,
    }
    values.update(overrides)
    return CalibrationDiagnostics(**values)


def test_quality_score_is_bounded() -> None:
    config = QualityConfig()

    assert 0.0 <= score_quality(diagnostics(), config) <= 1.0
    assert score_quality(diagnostics(inlier_ratio=5.0), config) <= 1.0
    assert score_quality(diagnostics(inlier_ratio=-2.0), config) >= 0.0


def test_quality_decreases_as_valid_calibration_ages() -> None:
    config = QualityConfig(max_prediction_age_frames=10)

    fresh = score_quality(diagnostics(last_valid_age_frames=0), config)
    aged = score_quality(diagnostics(last_valid_age_frames=5), config)
    expired = score_quality(diagnostics(last_valid_age_frames=10), config)

    assert fresh > aged > expired


def test_poor_reprojection_and_coverage_reduce_quality() -> None:
    config = QualityConfig(
        max_reprojection_error=5.0,
        target_spatial_coverage=0.5,
    )

    good = score_quality(diagnostics(), config)
    poor = score_quality(
        diagnostics(reprojection_error=20.0, spatial_coverage=0.01),
        config,
    )

    assert good > poor
