from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def summarize_stage_latencies(
    seconds_by_stage: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for stage, samples in seconds_by_stage.items():
        values = np.asarray(samples, dtype=np.float64) * 1000.0
        if len(values) == 0:
            result[stage] = {
                "sample_count": 0,
                "mean_ms": None,
                "p50_ms": None,
                "p95_ms": None,
            }
        else:
            result[stage] = {
                "sample_count": int(len(values)),
                "mean_ms": float(np.mean(values)),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
            }
    return result


def dropped_frames(*, expected_frames: int | None, processed_frames: int) -> int | None:
    if expected_frames is None:
        return None
    if expected_frames < 0 or processed_frames < 0:
        raise ValueError("frame counts must be non-negative")
    return max(0, expected_frames - processed_frames)
