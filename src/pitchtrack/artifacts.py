from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pitchtrack.schemas import FrameResult
from pitchtrack.video import VideoMetadata


class ArtifactWriter:
    def __init__(
        self,
        output_dir: str | Path,
        metadata: VideoMetadata,
        rendered_size: tuple[int, int],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tracks = (self.output_dir / "tracks.jsonl").open("w", encoding="utf-8")
        self._calibration = (self.output_dir / "calibration.jsonl").open("w", encoding="utf-8")
        self._video = cv2.VideoWriter(
            str(self.output_dir / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            rendered_size,
        )
        if not self._video.isOpened():
            self._tracks.close()
            self._calibration.close()
            raise RuntimeError("Could not create annotated.mp4 with the mp4v codec")
        self.frames_written = 0

    def write(self, result: FrameResult, rendered_bgr: np.ndarray) -> None:
        self._tracks.write(json.dumps(result.to_dict(), allow_nan=False) + "\n")
        calibration = {
            "frame_index": result.frame_index,
            "timestamp_s": result.timestamp_s,
            **result.calibration.to_dict(),
        }
        self._calibration.write(json.dumps(calibration, allow_nan=False) + "\n")
        self._tracks.flush()
        self._calibration.flush()
        self._video.write(rendered_bgr)
        self.frames_written += 1

    def write_summary(
        self,
        run_metadata: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    def close(self) -> None:
        self._tracks.close()
        self._calibration.close()
        self._video.release()

    def __enter__(self) -> "ArtifactWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
