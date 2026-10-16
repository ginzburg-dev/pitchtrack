import numpy as np

from pitchtrack.calibration.geometry import (
    HomographyConfig,
    estimate_homography,
    project_points,
)
from pitchtrack.schemas import CalibrationStatus


IMAGE_POINTS = np.array(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 300.0], [100.0, 300.0]],
    dtype=np.float64,
)
PITCH_POINTS = np.array(
    [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
    dtype=np.float64,
)


def test_estimate_homography_recovers_known_image_to_pitch_mapping() -> None:
    config = HomographyConfig(
        image_size_px=(600, 400),
        min_spatial_coverage=0.2,
        max_reprojection_error=0.01,
    )

    result = estimate_homography(IMAGE_POINTS, PITCH_POINTS, config)
    projected = project_points(IMAGE_POINTS, result.H_img_to_pitch)

    assert result.status is CalibrationStatus.VALID
    assert result.H_img_to_pitch is not None
    np.testing.assert_allclose(projected, PITCH_POINTS, atol=1e-5)
    assert result.diagnostics.reprojection_error < 1e-5
    assert result.diagnostics.reprojection_error_unit == "m"
    assert result.diagnostics.inlier_ratio == 1.0


def test_collinear_correspondences_return_invalid_without_identity_fallback() -> None:
    collinear = np.array(
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        dtype=np.float64,
    )

    result = estimate_homography(
        collinear,
        collinear,
        HomographyConfig(image_size_px=(100, 100)),
    )

    assert result.status is CalibrationStatus.INVALID
    assert result.H_img_to_pitch is None
    assert result.reason == "degenerate_correspondences"


def test_correspondences_with_tiny_image_coverage_are_invalid() -> None:
    image_points = np.array(
        [[10.0, 10.0], [12.0, 10.0], [12.0, 12.0], [10.0, 12.0]],
        dtype=np.float64,
    )
    pitch_points = np.array(
        [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]],
        dtype=np.float64,
    )

    result = estimate_homography(
        image_points,
        pitch_points,
        HomographyConfig(image_size_px=(1000, 1000), min_spatial_coverage=0.01),
    )

    assert result.status is CalibrationStatus.INVALID
    assert result.H_img_to_pitch is None
    assert result.reason == "insufficient_spatial_coverage"


def test_project_points_rejects_non_finite_homography() -> None:
    bad_h = np.eye(3, dtype=np.float64)
    bad_h[0, 0] = np.nan

    try:
        project_points(np.array([[1.0, 2.0]]), bad_h)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite homography must be rejected")

