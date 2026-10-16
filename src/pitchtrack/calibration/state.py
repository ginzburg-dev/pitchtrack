from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibrationStateConfig:
    max_prediction_age_frames: int = 15
    prediction_confidence_decay: float = 0.9
    reinitialize_interval_frames: int = 30
    max_anchor_disagreement: float = 5.0

    def __post_init__(self) -> None:
        if self.max_prediction_age_frames <= 0:
            raise ValueError("max_prediction_age_frames must be positive")
        if not 0.0 < self.prediction_confidence_decay < 1.0:
            raise ValueError("prediction_confidence_decay must be strictly between 0 and 1")
        if self.reinitialize_interval_frames <= 0:
            raise ValueError("reinitialize_interval_frames must be positive")
        if self.max_anchor_disagreement <= 0:
            raise ValueError("max_anchor_disagreement must be positive")
