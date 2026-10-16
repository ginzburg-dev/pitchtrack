from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from pitchtrack.schemas import Detection


@runtime_checkable
class PlayerDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> Sequence[Detection]: ...
