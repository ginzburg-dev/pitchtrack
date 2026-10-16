from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ReprojectionSample:
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value < 0:
            raise ValueError("reprojection error must be finite and non-negative")
        if not self.unit:
            raise ValueError("reprojection unit is required")


def tolerant_jaccard(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    tolerance_px: int = 5,
) -> float | None:
    prediction = np.asarray(predicted, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError("predicted and target masks must have the same shape")
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")
    predicted_count = int(np.count_nonzero(prediction))
    target_count = int(np.count_nonzero(truth))
    if predicted_count == 0 and target_count == 0:
        return None
    kernel_size = 2 * tolerance_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    target_region = cv2.dilate(truth.astype(np.uint8), kernel).astype(bool)
    prediction_region = cv2.dilate(prediction.astype(np.uint8), kernel).astype(bool)
    precision = (
        float(np.count_nonzero(prediction & target_region)) / predicted_count
        if predicted_count
        else 0.0
    )
    recall = (
        float(np.count_nonzero(truth & prediction_region)) / target_count
        if target_count
        else 0.0
    )
    return min(precision, recall)


def completeness(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return float(sum(bool(value) for value in values) / len(values))


def mean_reprojection_error(
    samples: Sequence[ReprojectionSample],
) -> dict[str, float | str | int | None]:
    if not samples:
        return {"value": None, "unit": None, "sample_count": 0}
    units = {sample.unit for sample in samples}
    if len(units) != 1:
        raise ValueError("mixed reprojection units are not comparable")
    return {
        "value": float(np.mean([sample.value for sample in samples])),
        "unit": next(iter(units)),
        "sample_count": len(samples),
    }
