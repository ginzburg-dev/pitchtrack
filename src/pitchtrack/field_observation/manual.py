from __future__ import annotations

import numpy as np

from pitchtrack.field_observation.base import FieldObservation


class ManualFieldObservationProvider:
    def __init__(self, observation: FieldObservation) -> None:
        self._observation = observation

    def observe(self, frame_bgr: np.ndarray) -> FieldObservation:
        del frame_bgr
        return self._observation

