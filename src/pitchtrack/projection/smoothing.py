from __future__ import annotations

from pitchtrack.schemas import CalibrationStatus


class PitchTrackSmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._state: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(
        self,
        track_id: int | None,
        raw_xy: tuple[float, float] | None,
        calibration_status: CalibrationStatus,
    ) -> tuple[float, float] | None:
        if calibration_status is CalibrationStatus.INVALID or raw_xy is None:
            if track_id is not None:
                self._state.pop(track_id, None)
            return None
        if track_id is None:
            return raw_xy
        previous = self._state.get(track_id)
        if previous is None:
            smoothed = (float(raw_xy[0]), float(raw_xy[1]))
        else:
            smoothed = (
                self.alpha * raw_xy[0] + (1.0 - self.alpha) * previous[0],
                self.alpha * raw_xy[1] + (1.0 - self.alpha) * previous[1],
            )
        self._state[track_id] = smoothed
        return smoothed
