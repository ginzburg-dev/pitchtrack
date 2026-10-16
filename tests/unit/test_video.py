from pathlib import Path

import cv2
import numpy as np
import pytest

from pitchtrack.video import VideoOpenError, VideoReader


def write_test_video(path: Path, frame_count: int = 5, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_reader_emits_ordered_frames_and_timestamps(tmp_path: Path) -> None:
    video_path = tmp_path / "ordered.mp4"
    write_test_video(video_path)

    with VideoReader(video_path) as reader:
        frames = list(reader)
        metadata = reader.metadata

    assert metadata.fps == pytest.approx(10.0)
    assert metadata.width == 64
    assert metadata.height == 48
    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3, 4]
    assert [frame.timestamp_s for frame in frames] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_video_reader_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        VideoReader(tmp_path / "missing.mp4")


def test_video_reader_rejects_undecodable_file(tmp_path: Path) -> None:
    bad_path = tmp_path / "truncated.mp4"
    bad_path.write_bytes(b"not an mp4")

    with pytest.raises(VideoOpenError, match="Could not open video"):
        VideoReader(bad_path)
