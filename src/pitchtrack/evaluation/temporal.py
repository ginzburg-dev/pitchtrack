from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pitchtrack.schemas import CalibrationStatus


def status_coverage(
    statuses: Sequence[CalibrationStatus | str],
) -> dict[str, float | None]:
    if not statuses:
        return {status.value: None for status in CalibrationStatus}
    normalized = [CalibrationStatus(status) for status in statuses]
    return {
        status.value: normalized.count(status) / len(normalized) for status in CalibrationStatus
    }


def calibration_recovery(
    statuses: Sequence[CalibrationStatus | str],
) -> dict[str, object]:
    normalized = [CalibrationStatus(status) for status in statuses]
    recoveries: list[int] = []
    failure_start: int | None = None
    for index, status in enumerate(normalized):
        if status is CalibrationStatus.INVALID and failure_start is None:
            failure_start = index
        elif status is CalibrationStatus.VALID and failure_start is not None:
            recoveries.append(index - failure_start)
            failure_start = None
    return {
        "recovery_frames": recoveries,
        "mean_recovery_frames": (float(np.mean(recoveries)) if recoveries else None),
        "unresolved_failures": int(failure_start is not None),
    }


def trajectory_jitter(points: Sequence[tuple[float, float] | None]) -> float | None:
    if any(point is None for point in points) or len(points) < 3:
        return None
    array = np.asarray(points, dtype=np.float64)
    acceleration = np.diff(array, n=2, axis=0)
    return float(np.mean(np.linalg.norm(acceleration, axis=1)))


def response_lag_frames(
    raw: Sequence[tuple[float, float]],
    smoothed: Sequence[tuple[float, float] | None],
    *,
    movement_threshold: float,
) -> int | None:
    if len(raw) != len(smoothed) or not raw:
        return None

    def crossing(points: Sequence[tuple[float, float] | None]) -> int | None:
        if points[0] is None:
            return None
        origin = np.asarray(points[0], dtype=np.float64)
        for index, point in enumerate(points):
            if (
                point is not None
                and np.linalg.norm(np.asarray(point) - origin) >= movement_threshold
            ):
                return index
        return None

    raw_crossing = crossing(raw)
    smooth_crossing = crossing(smoothed)
    if raw_crossing is None or smooth_crossing is None:
        return None
    return max(0, smooth_crossing - raw_crossing)
