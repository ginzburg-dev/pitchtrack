from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from pitchtrack.field_observation.base import FieldObservation


class DeepLabFieldObserver:
    """Adapter for an exported SoccerNet DeepLab landmark predictor."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        predictor: Callable[[np.ndarray], Mapping[str, Any]] | None = None,
    ) -> None:
        self._predictor = predictor or _load_torchscript(model_path, "DeepLab")

    def observe(self, frame_bgr: np.ndarray) -> FieldObservation:
        return _observation(self._predictor(frame_bgr), "baseline_deeplab")


def _load_torchscript(model_path: str | Path | None, label: str):
    if model_path is None or not Path(model_path).is_file():
        raise FileNotFoundError(
            f"{label} local weights not found: {model_path!s}. "
            "Provide an exported local checkpoint; PitchTrack does not download weights."
        )
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise RuntimeError("PyTorch is optional; install the 'models' extra.") from error
    model = torch.jit.load(str(Path(model_path).resolve()), map_location="cpu")
    model.eval()

    def predict(frame: np.ndarray):
        tensor = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            return model(tensor.unsqueeze(0))

    return predict


def _observation(raw: Mapping[str, Any], source: str) -> FieldObservation:
    return FieldObservation(
        image_points_px=np.asarray(raw["image_points_px"], dtype=np.float64),
        pitch_points_m=np.asarray(raw["pitch_points_m"], dtype=np.float64),
        confidences=np.asarray(raw["confidences"], dtype=np.float64),
        source=source,
    )
