import cv2
import numpy as np

from pitchtrack.calibration.flow import (
    FlowCorrespondences,
    OpticalFlowConfig,
    estimate_interframe_motion,
    track_static_features,
)


def textured_frame(width: int = 320, height: int = 240) -> np.ndarray:
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_optical_flow_tracks_pan_and_excludes_moving_mask() -> None:
    previous = textured_frame()
    affine = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    current = cv2.warpAffine(previous, affine, (320, 240))
    exclusion = np.zeros((240, 320), dtype=np.uint8)
    exclusion[0:100, 0:120] = 255

    flow = track_static_features(
        previous,
        current,
        exclusion,
        OpticalFlowConfig(max_corners=300, min_points=12, fb_max_error=1.0),
    )

    assert flow.count >= 12
    median_motion = np.median(flow.current_points_px - flow.previous_points_px, axis=0)
    np.testing.assert_allclose(median_motion, [4.0, 2.0], atol=0.35)
    assert not np.any(
        (flow.previous_points_px[:, 0] < 120)
        & (flow.previous_points_px[:, 1] < 100)
    )


def test_ransac_interframe_motion_rejects_outliers() -> None:
    xs, ys = np.meshgrid(np.linspace(20, 300, 8), np.linspace(20, 220, 6))
    previous = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    current = previous + np.array([5.0, -3.0])
    current[-8:] += np.array([60.0, 40.0])
    correspondences = FlowCorrespondences(previous, current, np.zeros(len(previous)))

    motion = estimate_interframe_motion(
        correspondences,
        frame_size_px=(320, 240),
        config=OpticalFlowConfig(min_points=12, min_inlier_ratio=0.6),
    )

    assert motion.valid
    assert motion.inlier_ratio > 0.7
    homogeneous = np.column_stack([current[:-8], np.ones(len(current) - 8)])
    mapped = (motion.H_current_to_previous @ homogeneous.T).T
    mapped = mapped[:, :2] / mapped[:, 2:]
    np.testing.assert_allclose(mapped, previous[:-8], atol=0.15)


def test_spatially_clustered_flow_is_invalid() -> None:
    previous = np.array(
        [[10.0, 10.0], [12.0, 10.0], [14.0, 10.0], [10.0, 12.0], [12.0, 12.0], [14.0, 12.0]],
        dtype=np.float64,
    )
    current = previous + np.array([2.0, 1.0])

    motion = estimate_interframe_motion(
        FlowCorrespondences(previous, current, np.zeros(len(previous))),
        frame_size_px=(1000, 1000),
        config=OpticalFlowConfig(min_points=4, min_spatial_coverage=0.01),
    )

    assert not motion.valid
    assert motion.H_current_to_previous is None
    assert motion.reason == "insufficient_spatial_coverage"

