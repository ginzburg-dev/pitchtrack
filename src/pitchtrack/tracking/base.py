from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from pitchtrack.schemas import Detection, TrackObservation


@runtime_checkable
class ImageSpaceTracker(Protocol):
    def update(
        self, detections: Sequence[Detection], frame_bgr: np.ndarray
    ) -> Sequence[TrackObservation]: ...
