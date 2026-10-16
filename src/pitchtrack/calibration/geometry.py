from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pitchtrack.calibration.quality import QualityConfig, score_quality
from pitchtrack.schemas import (
    CalibrationDiagnostics,
    CalibrationResult,
    CalibrationStatus,
)


@dataclass(frozen=True, slots=True)
class HomographyConfig:
    image_size_px: tuple[int, int]
    ransac_reprojection_threshold: float = 3.0
    min_inliers: int = 4
    min_inlier_ratio: float = 0.5
    min_spatial_coverage: float = 0.01
    max_reprojection_error: float = 5.0
    reprojection_error_unit: str = "m"
    quality: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self) -> None:
        width, height = self.image_size_px
        if width <= 0 or height <= 0:
            raise ValueError("image_size_px must be positive")
        if self.ransac_reprojection_threshold <= 0:
            raise ValueError("ransac_reprojection_threshold must be positive")
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be at least four")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between 0 and 1")


def _points_array(points: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _is_degenerate(points: np.ndarray) -> bool:
    if len(points) < 4:
        return True
    centered = points - np.mean(points, axis=0, keepdims=True)
    return np.linalg.matrix_rank(centered, tol=1e-9) < 2


def _spatial_coverage(points: np.ndarray, image_size_px: tuple[int, int]) -> float:
    hull = cv2.convexHull(points.astype(np.float32))
    area = float(cv2.contourArea(hull))
    width, height = image_size_px
    return area / float(width * height)


def project_points(points_px: np.ndarray, H_img_to_pitch: np.ndarray) -> np.ndarray:
    points = _points_array(points_px, "points_px")
    matrix = np.asarray(H_img_to_pitch, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("H_img_to_pitch must be a finite 3x3 matrix")
    projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        raise ValueError("projected points are not finite")
    return projected


def estimate_homography(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    config: HomographyConfig,
) -> CalibrationResult:
    image = _points_array(image_points, "image_points")
    pitch = _points_array(pitch_points, "pitch_points")
    if len(image) != len(pitch):
        raise ValueError("image_points and pitch_points must have equal length")
    if _is_degenerate(image) or _is_degenerate(pitch):
        return CalibrationResult.invalid("degenerate_correspondences")

    coverage = _spatial_coverage(image, config.image_size_px)
    if coverage < config.min_spatial_coverage:
        diagnostics = CalibrationDiagnostics(spatial_coverage=coverage)
        return CalibrationResult.invalid("insufficient_spatial_coverage", diagnostics)

    matrix, mask = cv2.findHomography(
        image,
        pitch,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reprojection_threshold,
    )
    if matrix is None or mask is None or not np.all(np.isfinite(matrix)):
        return CalibrationResult.invalid("homography_estimation_failed")
    if abs(float(matrix[2, 2])) < 1e-12:
        return CalibrationResult.invalid("homography_normalization_failed")
    matrix = matrix / matrix[2, 2]

    inliers = mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / len(image)
    projected = project_points(image[inliers], matrix)
    errors = np.linalg.norm(projected - pitch[inliers], axis=1)
    reprojection_error = float(np.mean(errors)) if len(errors) else float("inf")
    diagnostics = CalibrationDiagnostics(
        inlier_ratio=inlier_ratio,
        reprojection_error=reprojection_error,
        reprojection_error_unit=config.reprojection_error_unit,
        spatial_coverage=coverage,
        temporal_consistency=1.0,
        last_valid_age_frames=0,
    )
    if inlier_count < config.min_inliers or inlier_ratio < config.min_inlier_ratio:
        return CalibrationResult.invalid("insufficient_ransac_inliers", diagnostics)
    if reprojection_error > config.max_reprojection_error:
        return CalibrationResult.invalid("excessive_reprojection_error", diagnostics)

    quality_config = QualityConfig(
        max_reprojection_error=config.max_reprojection_error,
        target_spatial_coverage=config.quality.target_spatial_coverage,
        max_prediction_age_frames=config.quality.max_prediction_age_frames,
    )
    return CalibrationResult(
        status=CalibrationStatus.VALID,
        H_img_to_pitch=matrix,
        confidence=score_quality(diagnostics, quality_config),
        diagnostics=diagnostics,
    )
