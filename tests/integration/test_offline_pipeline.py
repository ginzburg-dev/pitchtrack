import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pitchtrack.pipeline import build_fixture_pipeline
from pitchtrack.projection.projector import PitchProjector
from pitchtrack.schemas import CalibrationResult, CalibrationStatus, Detection, TrackObservation
from pitchtrack.testing.fixture import create_demo_fixture
from pitchtrack.video import VideoDecodeError, VideoFrame, VideoMetadata


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_offline_pipeline_writes_synchronized_artifacts(tmp_path: Path) -> None:
    fixture = create_demo_fixture(tmp_path / "fixture", frame_count=6, fps=10.0)
    output_dir = tmp_path / "run"
    pipeline = build_fixture_pipeline(fixture.config_path)

    summary = pipeline.run(fixture.video_path, output_dir)

    assert summary.frames_processed == 6
    expected = {
        "annotated.mp4",
        "tracks.jsonl",
        "calibration.jsonl",
        "run_metadata.json",
        "metrics.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    tracks = read_jsonl(output_dir / "tracks.jsonl")
    calibrations = read_jsonl(output_dir / "calibration.jsonl")
    assert len(tracks) == len(calibrations) == 6
    assert [record["frame_index"] for record in tracks] == list(range(6))
    assert [record["timestamp_s"] for record in tracks] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert all(record["tracks"][0]["track_id"] == 1 for record in tracks)
    assert all(record["tracks"][0]["field_xy_m"] is not None for record in tracks)
    assert all(record["tracks"][0]["smoothed_field_xy_m"] is not None for record in tracks)

    capture = cv2.VideoCapture(str(output_dir / "annotated.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    capture.release()

    metadata = json.loads((output_dir / "run_metadata.json").read_text("utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text("utf-8"))
    assert metadata["video"]["fps"] == 10.0
    assert metadata["frames_processed"] == 6
    assert metadata["source_kind"] == "fixture"
    assert metrics["frames_processed"] == 6
    assert metrics["calibration_status_counts"] == {"valid": 6}
    assert metrics["dropped_frames"] == 0
    assert metrics["stage_latency_ms"]["detect"]["sample_count"] == 6
    assert metrics["stage_latency_ms"]["calibrate"]["p95_ms"] is not None


def test_invalid_calibration_preserves_track_but_nulls_pitch_coordinates(
    tmp_path: Path,
) -> None:
    fixture = create_demo_fixture(
        tmp_path / "invalid-fixture",
        frame_count=2,
        calibration_status=CalibrationStatus.INVALID,
    )
    output_dir = tmp_path / "invalid-run"

    build_fixture_pipeline(fixture.config_path).run(fixture.video_path, output_dir)

    tracks = read_jsonl(output_dir / "tracks.jsonl")
    assert [record["tracks"][0]["track_id"] for record in tracks] == [1, 1]
    assert [record["tracks"][0]["field_xy_m"] for record in tracks] == [None, None]
    assert [record["tracks"][0]["smoothed_field_xy_m"] for record in tracks] == [
        None,
        None,
    ]
    assert [record["calibration"]["status"] for record in tracks] == [
        "invalid",
        "invalid",
    ]


def test_pitch_projector_never_uses_geometry_marked_invalid() -> None:
    track = TrackObservation(
        track_id=3,
        detection=Detection((10.0, 10.0, 20.0, 30.0), 0.9),
    )

    projected = PitchProjector().project(
        track,
        CalibrationResult.invalid("lost_field"),
    )

    assert projected.track_id == 3
    assert projected.field_xy_m is None


def test_decode_failure_keeps_only_complete_frame_records(tmp_path: Path) -> None:
    fixture = create_demo_fixture(tmp_path / "truncated-fixture", frame_count=3)
    output_dir = tmp_path / "truncated-run"

    class TruncatingReader:
        metadata = VideoMetadata(
            fps=10.0,
            width=320,
            height=180,
            declared_frame_count=3,
        )

        def __enter__(self) -> "TruncatingReader":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self):
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            yield VideoFrame(0, 0.0, image)
            yield VideoFrame(1, 0.1, image)
            raise VideoDecodeError("synthetic truncated stream")

    pipeline = build_fixture_pipeline(
        fixture.config_path,
        reader_factory=lambda _: TruncatingReader(),
    )

    with pytest.raises(VideoDecodeError, match="truncated"):
        pipeline.run(fixture.video_path, output_dir)

    assert len(read_jsonl(output_dir / "tracks.jsonl")) == 2
    assert len(read_jsonl(output_dir / "calibration.jsonl")) == 2
    assert not (output_dir / "run_metadata.json").exists()
    assert not (output_dir / "metrics.json").exists()
