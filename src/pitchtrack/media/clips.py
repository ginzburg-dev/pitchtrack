from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameWindow:
    start: int
    stop: int


def select_dynamic_window(
    records: list[dict[str, Any]], *, fps: float, duration_s: float = 6.0
) -> FrameWindow:
    if not records:
        raise ValueError("calibration records are empty")
    scores: list[tuple[float, int]] = []
    previous: np.ndarray | None = None
    for index, record in enumerate(records):
        raw = record.get("H_img_to_pitch")
        current = np.asarray(raw, dtype=np.float64) if raw is not None else None
        if current is not None and current.shape == (3, 3) and previous is not None:
            current = current / current[2, 2]
            scores.append((float(np.linalg.norm(current - previous)), index))
        if current is not None and current.shape == (3, 3):
            previous = current / current[2, 2]
    center = max(scores, default=(0.0, 0))[1]
    return _centered_window(center, len(records), fps, duration_s)


def select_failure_window(
    records: list[dict[str, Any]], *, fps: float, duration_s: float = 6.0
) -> FrameWindow | None:
    for index, record in enumerate(records):
        if record.get("status") == "invalid":
            return _centered_window(index, len(records), fps, duration_s)
    return None


def _centered_window(center: int, total: int, fps: float, duration_s: float) -> FrameWindow:
    count = max(1, min(total, int(round(fps * duration_s))))
    start = min(max(0, center - count // 2), total - count)
    return FrameWindow(start, start + count)
