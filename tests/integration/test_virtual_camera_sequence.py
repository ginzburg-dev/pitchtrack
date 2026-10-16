import json
from pathlib import Path

import numpy as np
import yaml

from pitchtrack.calibration.geometry import project_points
from pitchtrack.pipeline import build_fixture_pipeline
from pitchtrack.testing.fixture import create_demo_fixture


def test_virtual_camera_fixture_emits_changing_accurate_homographies(
    tmp_path: Path,
) -> None:
    fixture = create_demo_fixture(
        tmp_path / "virtual-camera",
        frame_count=6,
        camera_pan_px_per_frame=2.0,
    )
    output = tmp_path / "run"

    pipeline = build_fixture_pipeline(
        fixture.config_path,
        calibration_mode="hybrid_temporal",
    )
    pipeline.run(fixture.video_path, output)

    config = yaml.safe_load(fixture.config_path.read_text("utf-8"))
    expected_h = [np.asarray(value) for value in config["ground_truth_H_img_to_pitch"]]
    actual_records = [
        json.loads(line)
        for line in (output / "calibration.jsonl").read_text("utf-8").splitlines()
    ]
    actual_h = [np.asarray(record["H_img_to_pitch"]) for record in actual_records]
    anchors = np.array([[60.0, 60.0], [160.0, 100.0], [260.0, 140.0]])
    errors = [
        np.mean(
            np.linalg.norm(
                project_points(anchors, actual) - project_points(anchors, expected),
                axis=1,
            )
        )
        for actual, expected in zip(actual_h, expected_h, strict=True)
    ]

    assert actual_records[0]["status"] == "valid"
    assert {record["status"] for record in actual_records[1:]} == {"predicted"}
    assert not np.allclose(actual_h[0], actual_h[-1])
    assert max(errors) < 0.8
