from __future__ import annotations

from dataclasses import dataclass

from pitchtrack.schemas import CalibrationDiagnostics


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class QualityConfig:
    max_reprojection_error: float = 5.0
    target_spatial_coverage: float = 0.25
    max_prediction_age_frames: int = 15

    def __post_init__(self) -> None:
        if self.max_reprojection_error <= 0:
            raise ValueError("max_reprojection_error must be positive")
        if self.target_spatial_coverage <= 0:
            raise ValueError("target_spatial_coverage must be positive")
        if self.max_prediction_age_frames <= 0:
            raise ValueError("max_prediction_age_frames must be positive")


def score_quality(
    diagnostics: CalibrationDiagnostics,
    config: QualityConfig,
) -> float:
    inliers = _clamp01(diagnostics.inlier_ratio)
    reprojection = _clamp01(1.0 - diagnostics.reprojection_error / config.max_reprojection_error)
    coverage = _clamp01(diagnostics.spatial_coverage / config.target_spatial_coverage)
    temporal = _clamp01(diagnostics.temporal_consistency)
    age = _clamp01(1.0 - diagnostics.last_valid_age_frames / config.max_prediction_age_frames)
    return _clamp01(
        0.25 * inliers + 0.25 * reprojection + 0.20 * coverage + 0.20 * temporal + 0.10 * age
    )
