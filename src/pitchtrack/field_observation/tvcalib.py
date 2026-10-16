from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pitchtrack.field_observation.base import FieldObservation
from pitchtrack.field_observation.deeplab import _load_torchscript, _observation


class TVCalibFieldObserver:
    """Adapter for an exported TVCalib predictor using the common landmark contract."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        predictor: Callable[[np.ndarray], Mapping[str, Any]] | None = None,
    ) -> None:
        self._predictor = predictor or _load_torchscript(model_path, "TVCalib")

    def observe(self, frame_bgr: np.ndarray) -> FieldObservation:
        return _observation(self._predictor(frame_bgr), "baseline_tvcalib")
