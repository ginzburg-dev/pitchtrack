import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pitchtrack.media import build_media_assets
from pitchtrack.media.clips import select_dynamic_window, select_failure_window
from pitchtrack.media.encode import enforce_asset_size


def calibration(index: int, x: float, status: str = "valid") -> dict:
    return {
        "frame_index": index,
        "status": status,
        "H_img_to_pitch": None if status == "invalid" else [[1, 0, x], [0, 1, 0], [0, 0, 1]],
    }


def make_run(path: Path, *, source_kind: str, include_failure: bool = True) -> None:
    path.mkdir()
    writer = cv2.VideoWriter(
        str(path / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 90),
    )
    calibrations = []
    tracks = []
    for index in range(20):
        frame = np.full((90, 160, 3), (index * 7) % 255, dtype=np.uint8)
        cv2.putText(frame, str(index), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
        status = "invalid" if include_failure and index in (12, 13) else "valid"
        calibrations.append(calibration(index, 20.0 if index >= 8 else 0.0, status))
        tracks.append({"frame_index": index, "tracks": [], "calibration": calibrations[-1]})
    writer.release()
    (path / "calibration.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in calibrations), encoding="utf-8"
    )
    (path / "tracks.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in tracks), encoding="utf-8"
    )
    (path / "run_metadata.json").write_text(
        json.dumps({"source_kind": source_kind, "frames_processed": 20}),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"frames_processed": 20, "throughput_fps": 12.5}),
        encoding="utf-8",
    )


def test_window_selection_finds_camera_change_and_invalid_transition() -> None:
    records = [
        calibration(i, 15.0 if i >= 6 else 0.0, "invalid" if i == 9 else "valid") for i in range(12)
    ]

    dynamic = select_dynamic_window(records, fps=2.0, duration_s=3.0)
    failure = select_failure_window(records, fps=2.0, duration_s=3.0)

    assert dynamic.start <= 6 < dynamic.stop
    assert failure is not None
    assert failure.start <= 9 < failure.stop


def test_failure_window_is_unavailable_without_invalid_geometry() -> None:
    assert select_failure_window([calibration(i, 0.0) for i in range(5)], fps=2.0) is None


def test_builder_writes_looping_assets_and_explicit_provenance(tmp_path: Path) -> None:
    run = tmp_path / "run"
    output = tmp_path / "assets"
    make_run(run, source_kind="annotation-reference")

    manifest = build_media_assets(run, output, "annotation-reference")

    assert (output / "demo-overview.gif").is_file()
    assert (output / "dynamic-calibration.gif").is_file()
    assert (output / "failure-handling.gif").is_file()
    assert (output / "full-demo.mp4").is_file()
    assert manifest.source_kind == "annotation-reference"
    payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert payload["source_kind"] == "annotation-reference"
    assert all(item["sha256"] for item in payload["assets"])


def test_builder_rejects_a_false_model_output_label(tmp_path: Path) -> None:
    run = tmp_path / "run"
    make_run(run, source_kind="annotation-reference")

    with pytest.raises(ValueError, match="source_kind"):
        build_media_assets(run, tmp_path / "assets", "model")


def test_size_guard_rejects_oversized_asset(tmp_path: Path) -> None:
    asset = tmp_path / "large.bin"
    asset.write_bytes(b"x" * 11)

    with pytest.raises(ValueError, match="size limit"):
        enforce_asset_size(asset, max_bytes=10)
