import ast
from pathlib import Path

import numpy as np
import pytest

from pitchtrack.calibration.geometry import HomographyConfig
from pitchtrack.field_observation.base import FieldObservation
from pitchtrack.field_observation.deeplab import DeepLabFieldObserver
from pitchtrack.field_observation.tvcalib import TVCalibFieldObserver
from pitchtrack.pipeline import FieldCalibrationSource, moving_object_mask
from pitchtrack.schemas import CalibrationStatus, Detection


def test_field_observers_normalize_a_shared_prediction_contract() -> None:
    prediction = {
        "image_points_px": [[10, 20], [30, 40], [50, 60], [70, 80]],
        "pitch_points_m": [[0, 0], [10, 0], [10, 20], [0, 20]],
        "confidences": [0.9, 0.8, 0.7, 0.6],
    }
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    deep = DeepLabFieldObserver(predictor=lambda _: prediction).observe(frame)
    tv = TVCalibFieldObserver(predictor=lambda _: prediction).observe(frame)

    assert deep.source == "baseline_deeplab"
    assert tv.source == "baseline_tvcalib"
    np.testing.assert_allclose(deep.image_points_px, tv.image_points_px)


@pytest.mark.parametrize("observer", [DeepLabFieldObserver, TVCalibFieldObserver])
def test_optional_observer_requires_explicit_local_weights(observer, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="local weights"):
        observer(model_path=tmp_path / "missing.ckpt")


def test_moving_object_mask_covers_detection_boxes() -> None:
    mask = moving_object_mask((40, 60), [Detection((10, 5, 20, 30), 0.9)])

    assert mask.shape == (40, 60)
    assert mask[10, 15] == 255
    assert mask[0, 0] == 0


def test_source_imports_are_module_scoped() -> None:
    violations = []
    for path in (Path(__file__).parents[2] / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for container in ast.walk(tree):
            if not isinstance(container, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for node in ast.walk(container):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    violations.append(f"{path.name}:{node.lineno}")
    assert not violations, "nested imports: " + ", ".join(violations)


class FakeObserver:
    def __init__(self) -> None:
        self.calls = 0

    def observe(self, frame_bgr: np.ndarray) -> FieldObservation:
        self.calls += 1
        return FieldObservation(
            image_points_px=np.asarray([[0, 0], [99, 0], [99, 99], [0, 99]]),
            pitch_points_m=np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]]),
            confidences=np.ones(4),
            source="fake",
        )


@pytest.mark.parametrize("mode", ["baseline_deeplab", "baseline_tvcalib"])
def test_baseline_calibration_modes_estimate_geometry(mode: str) -> None:
    observer = FakeObserver()
    source = FieldCalibrationSource(
        mode,
        observer,
        HomographyConfig(image_size_px=(100, 100), min_spatial_coverage=0.1),
    )

    result = source.update(np.zeros((100, 100, 3), dtype=np.uint8), 0)

    assert result.status is CalibrationStatus.VALID
    assert observer.calls == 1


def test_hybrid_mode_reuses_last_geometry_between_semantic_observations() -> None:
    observer = FakeObserver()
    source = FieldCalibrationSource(
        "hybrid_temporal",
        observer,
        HomographyConfig(image_size_px=(100, 100), min_spatial_coverage=0.1),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = source.update(frame, 0)
    second = source.update(frame, 1)

    assert first.status is CalibrationStatus.VALID
    assert second.status in {CalibrationStatus.PREDICTED, CalibrationStatus.VALID}
    assert observer.calls == 1
