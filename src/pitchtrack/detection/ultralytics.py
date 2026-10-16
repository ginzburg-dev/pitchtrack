from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from pitchtrack.schemas import Detection


class UltralyticsPersonDetector:
    """Local-only Ultralytics adapter. It never resolves or downloads model names."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        predictor: Callable[[np.ndarray], Any] | None = None,
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.confidence = float(confidence)
        if predictor is not None:
            self._predictor = predictor
            return
        if model_path is None or not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Ultralytics local weights not found: {model_path!s}. "
                "Provide an existing .pt file; PitchTrack does not download weights."
            )
        try:
            YOLO = import_module("ultralytics").YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is optional. Install PitchTrack with the 'models' extra."
            ) from error
        model = YOLO(str(Path(model_path).resolve()))
        self._predictor = lambda frame: model.predict(
            source=frame, conf=self.confidence, classes=[0], device=device, verbose=False
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return [
            item
            for item in _normalize_predictions(self._predictor(frame_bgr))
            if item.class_name == "person" and item.confidence >= self.confidence
        ]


def _normalize_predictions(raw: Any) -> Iterable[Detection]:
    if isinstance(raw, (list, tuple)) and (not raw or isinstance(raw[0], Mapping)):
        for item in raw:
            yield Detection(
                tuple(item["bbox_xyxy_px"]),
                float(item["confidence"]),
                str(item.get("class_name", "person")),
            )
        return
    results = raw if isinstance(raw, (list, tuple)) else [raw]
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        names = getattr(result, "names", {})
        for bbox, score, class_id in zip(xyxy, confidences, classes, strict=True):
            class_name = names.get(int(class_id), str(class_id))
            yield Detection(tuple(float(value) for value in bbox), float(score), class_name)
