from pathlib import Path

import numpy as np
import pytest

from pitchtrack.detection.ultralytics import UltralyticsPersonDetector


class FakePredictor:
    def __call__(self, frame: np.ndarray):
        assert frame.shape == (20, 30, 3)
        return [
            {"bbox_xyxy_px": (1, 2, 11, 19), "confidence": 0.91, "class_name": "person"},
            {"bbox_xyxy_px": (5, 4, 8, 10), "confidence": 0.99, "class_name": "ball"},
            {"bbox_xyxy_px": (15, 2, 25, 18), "confidence": 0.2, "class_name": "person"},
        ]


def test_detector_normalizes_person_detections_and_applies_threshold() -> None:
    detector = UltralyticsPersonDetector(predictor=FakePredictor(), confidence=0.5)

    detections = detector.detect(np.zeros((20, 30, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].bbox_xyxy_px == (1.0, 2.0, 11.0, 19.0)
    assert detections[0].confidence == pytest.approx(0.91)


def test_detector_requires_local_weights_and_never_downloads(tmp_path: Path) -> None:
    missing = tmp_path / "yolo.pt"

    with pytest.raises(FileNotFoundError, match="local weights"):
        UltralyticsPersonDetector(model_path=missing)
