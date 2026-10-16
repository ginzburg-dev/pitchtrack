from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class OpticalFlowConfig:
    max_corners: int = 500
    quality_level: float = 0.01
    min_distance: float = 7.0
    block_size: int = 7
    window_size: tuple[int, int] = (21, 21)
    max_level: int = 3
    fb_max_error: float = 1.5
    min_points: int = 8
    ransac_reprojection_threshold: float = 2.0
    min_inlier_ratio: float = 0.5
    min_spatial_coverage: float = 0.05


@dataclass(frozen=True, slots=True)
class FlowCorrespondences:
    previous_points_px: np.ndarray
    current_points_px: np.ndarray
    forward_backward_error_px: np.ndarray

    def __post_init__(self) -> None:
        previous = np.asarray(self.previous_points_px, dtype=np.float64)
        current = np.asarray(self.current_points_px, dtype=np.float64)
        errors = np.asarray(self.forward_backward_error_px, dtype=np.float64)
        if previous.ndim != 2 or previous.shape[1:] != (2,):
            raise ValueError("previous_points_px must have shape (N, 2)")
        if current.shape != previous.shape or errors.shape != (len(previous),):
            raise ValueError("flow correspondence arrays must have matching lengths")
        if not all(np.all(np.isfinite(value)) for value in (previous, current, errors)):
            raise ValueError("flow correspondence arrays must be finite")
        object.__setattr__(self, "previous_points_px", previous.copy())
        object.__setattr__(self, "current_points_px", current.copy())
        object.__setattr__(self, "forward_backward_error_px", errors.copy())

    @property
    def count(self) -> int:
        return len(self.previous_points_px)

    @classmethod
    def empty(cls) -> "FlowCorrespondences":
        return cls(np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)))


@dataclass(frozen=True, slots=True)
class InterframeMotion:
    valid: bool
    H_current_to_previous: np.ndarray | None
    inlier_ratio: float
    reprojection_error_px: float
    spatial_coverage: float
    reason: str | None = None

    @classmethod
    def invalid(
        cls,
        reason: str,
        *,
        inlier_ratio: float = 0.0,
        reprojection_error_px: float = 0.0,
        spatial_coverage: float = 0.0,
    ) -> "InterframeMotion":
        return cls(
            False,
            None,
            inlier_ratio,
            reprojection_error_px,
            spatial_coverage,
            reason,
        )


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError("frame must be grayscale or BGR")


def track_static_features(
    previous: np.ndarray,
    current: np.ndarray,
    exclusion_mask: np.ndarray | None,
    config: OpticalFlowConfig,
) -> FlowCorrespondences:
    previous_gray = _gray(previous)
    current_gray = _gray(current)
    if previous_gray.shape != current_gray.shape:
        raise ValueError("previous and current frames must have equal size")
    allowed = np.full(previous_gray.shape, 255, dtype=np.uint8)
    excluded = np.zeros(previous_gray.shape, dtype=bool)
    if exclusion_mask is not None:
        if exclusion_mask.shape[:2] != previous_gray.shape:
            raise ValueError("exclusion_mask must match the frame size")
        excluded = np.asarray(exclusion_mask) > 0
        allowed[excluded] = 0

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        mask=allowed,
        blockSize=config.block_size,
    )
    if points is None:
        return FlowCorrespondences.empty()
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=config.window_size,
        maxLevel=config.max_level,
    )
    if current_points is None or forward_status is None:
        return FlowCorrespondences.empty()
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_points,
        None,
        winSize=config.window_size,
        maxLevel=config.max_level,
    )
    if backward_points is None or backward_status is None:
        return FlowCorrespondences.empty()

    previous_flat = points.reshape(-1, 2)
    current_flat = current_points.reshape(-1, 2)
    backward_flat = backward_points.reshape(-1, 2)
    errors = np.linalg.norm(previous_flat - backward_flat, axis=1)
    height, width = previous_gray.shape
    current_x = np.rint(current_flat[:, 0]).astype(int)
    current_y = np.rint(current_flat[:, 1]).astype(int)
    inside = (current_x >= 0) & (current_x < width) & (current_y >= 0) & (current_y < height)
    current_allowed = np.zeros(len(points), dtype=bool)
    current_allowed[inside] = ~excluded[current_y[inside], current_x[inside]]
    keep = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(errors)
        & (errors <= config.fb_max_error)
        & inside
        & current_allowed
    )
    return FlowCorrespondences(previous_flat[keep], current_flat[keep], errors[keep])


def _coverage(points: np.ndarray, frame_size_px: tuple[int, int]) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    width, height = frame_size_px
    return float(cv2.contourArea(hull)) / float(width * height)


def estimate_interframe_motion(
    correspondences: FlowCorrespondences,
    *,
    frame_size_px: tuple[int, int],
    config: OpticalFlowConfig,
) -> InterframeMotion:
    if correspondences.count < config.min_points:
        return InterframeMotion.invalid("insufficient_flow_points")
    coverage = _coverage(correspondences.previous_points_px, frame_size_px)
    if coverage < config.min_spatial_coverage:
        return InterframeMotion.invalid("insufficient_spatial_coverage", spatial_coverage=coverage)
    matrix, mask = cv2.findHomography(
        correspondences.current_points_px,
        correspondences.previous_points_px,
        cv2.RANSAC,
        config.ransac_reprojection_threshold,
    )
    if matrix is None or mask is None or not np.all(np.isfinite(matrix)):
        return InterframeMotion.invalid("motion_estimation_failed", spatial_coverage=coverage)
    if abs(float(matrix[2, 2])) < 1e-12:
        return InterframeMotion.invalid("motion_normalization_failed", spatial_coverage=coverage)
    matrix = matrix / matrix[2, 2]
    inliers = mask.reshape(-1).astype(bool)
    inlier_ratio = float(np.mean(inliers))
    current = correspondences.current_points_px[inliers]
    expected = correspondences.previous_points_px[inliers]
    homogeneous = np.column_stack([current, np.ones(len(current))])
    projected_h = (matrix @ homogeneous.T).T
    projected = projected_h[:, :2] / projected_h[:, 2:]
    reprojection_error = float(np.mean(np.linalg.norm(projected - expected, axis=1)))
    if inlier_ratio < config.min_inlier_ratio:
        return InterframeMotion.invalid(
            "insufficient_motion_inliers",
            inlier_ratio=inlier_ratio,
            reprojection_error_px=reprojection_error,
            spatial_coverage=coverage,
        )
    return InterframeMotion(
        True,
        matrix,
        inlier_ratio,
        reprojection_error,
        coverage,
    )
