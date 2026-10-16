from __future__ import annotations

import hashlib
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from pitchtrack import __version__
from pitchtrack.artifacts import ArtifactWriter
from pitchtrack.calibration.flow import OpticalFlowConfig
from pitchtrack.calibration.geometry import HomographyConfig, estimate_homography
from pitchtrack.calibration.hybrid import HybridCalibrationConfig, HybridTemporalCalibrator
from pitchtrack.calibration.state import CalibrationStateConfig
from pitchtrack.calibration.temporal import TemporalCalibrationConfig
from pitchtrack.field_observation.base import FieldObservation
from pitchtrack.projection.projector import PitchProjector
from pitchtrack.rendering.overlay import render_annotated_frame
from pitchtrack.schemas import (
    CalibrationResult,
    CalibrationStatus,
    Detection,
    FrameResult,
    TrackObservation,
)
from pitchtrack.testing.mocks import (
    HybridFixtureCalibrationSource,
    ScriptedDetector,
    StableSlotTracker,
    StaticCalibrator,
)
from pitchtrack.video import VideoReader


class PlayerDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> Sequence[Detection]: ...


class ImageSpaceTracker(Protocol):
    def update(
        self,
        detections: Sequence[Detection],
        frame_bgr: np.ndarray,
    ) -> Sequence[TrackObservation]: ...


class CalibrationSource(Protocol):
    def update(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        detections: Sequence[Detection] = (),
    ) -> CalibrationResult: ...


@dataclass(frozen=True, slots=True)
class RunSummary:
    frames_processed: int
    elapsed_s: float
    output_dir: Path


class OfflinePipeline:
    def __init__(
        self,
        detector: PlayerDetector,
        tracker: ImageSpaceTracker,
        calibrator: CalibrationSource,
        *,
        config_path: Path | None = None,
        reader_factory: Callable[[Path], VideoReader] = VideoReader,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.calibrator = calibrator
        self.projector = PitchProjector()
        self.config_path = config_path
        self.reader_factory = reader_factory

    def run(self, video_path: str | Path, output_dir: str | Path) -> RunSummary:
        started = time.perf_counter()
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        status_counts: Counter[str] = Counter()
        frames_processed = 0
        with self.reader_factory(video_path) as reader:
            rendered_width = reader.metadata.width + max(
                240, int(round(reader.metadata.height * 1.5))
            )
            with ArtifactWriter(
                output_dir,
                reader.metadata,
                (rendered_width, reader.metadata.height),
            ) as writer:
                for frame in reader:
                    detections = tuple(self.detector.detect(frame.image_bgr))
                    image_tracks = tuple(
                        self.tracker.update(detections, frame.image_bgr)
                    )
                    calibration = self.calibrator.update(
                        frame.image_bgr, frame.frame_index, detections
                    )
                    projected_tracks = tuple(
                        self.projector.project(track, calibration)
                        for track in image_tracks
                    )
                    result = FrameResult(
                        frame_index=frame.frame_index,
                        timestamp_s=frame.timestamp_s,
                        tracks=projected_tracks,
                        calibration=calibration,
                    )
                    rendered = render_annotated_frame(
                        frame.image_bgr, projected_tracks, calibration
                    )
                    writer.write(result, rendered)
                    frames_processed += 1
                    status_counts[calibration.status.value] += 1

                elapsed = time.perf_counter() - started
                metadata = {
                    "pitchtrack_version": __version__,
                    "python_version": platform.python_version(),
                    "opencv_version": cv2.__version__,
                    "numpy_version": np.__version__,
                    "command": " ".join(sys.argv),
                    "input_path": str(video_path.resolve()),
                    "input_sha256": _sha256(video_path),
                    "config_path": str(self.config_path.resolve()) if self.config_path else None,
                    "config_sha256": _sha256(self.config_path) if self.config_path else None,
                    "video": {
                        "fps": reader.metadata.fps,
                        "width": reader.metadata.width,
                        "height": reader.metadata.height,
                        "declared_frame_count": reader.metadata.declared_frame_count,
                    },
                    "frames_processed": frames_processed,
                    "completed": True,
                }
                metrics = {
                    "frames_processed": frames_processed,
                    "elapsed_s": elapsed,
                    "throughput_fps": frames_processed / elapsed if elapsed > 0 else 0.0,
                    "calibration_status_counts": dict(status_counts),
                }
                writer.write_summary(metadata, metrics)
        return RunSummary(frames_processed, time.perf_counter() - started, output_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixture_pipeline(
    config_path: str | Path,
    *,
    calibration_mode: str = "fixture",
    reader_factory: Callable[[Path], VideoReader] = VideoReader,
) -> OfflinePipeline:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("fixture") is not True:
        raise ValueError("Task 2 supports only deterministic fixture configs")
    detector = ScriptedDetector(config["detector"]["boxes_by_frame"])
    tracker = StableSlotTracker()
    calibration_config = config["calibration"]
    status = CalibrationStatus(calibration_config["status"])
    if calibration_mode == "hybrid_temporal":
        video = config["video"]
        homography = HomographyConfig(
            image_size_px=(int(video["width"]), int(video["height"])),
            min_spatial_coverage=0.1,
            max_reprojection_error=0.01,
        )
        temporal = TemporalCalibrationConfig(
            homography=homography,
            state=CalibrationStateConfig(
                max_prediction_age_frames=15,
                prediction_confidence_decay=0.9,
                reinitialize_interval_frames=30,
                max_anchor_disagreement=8.0,
            ),
        )
        hybrid = HybridTemporalCalibrator(
            HybridCalibrationConfig(
                temporal=temporal,
                flow=OpticalFlowConfig(
                    max_corners=400,
                    min_points=8,
                    min_spatial_coverage=0.05,
                ),
            )
        )
        first_observation = FieldObservation(
            image_points_px=np.asarray(calibration_config["image_points_px"]),
            pitch_points_m=np.asarray(calibration_config["pitch_points_m"]),
            confidences=np.ones(len(calibration_config["image_points_px"])),
            source="fixture-landmarks",
        )
        calibrator = HybridFixtureCalibrationSource(hybrid, first_observation)
    elif status is CalibrationStatus.INVALID:
        calibration = CalibrationResult.invalid(calibration_config["reason"])
        calibrator = StaticCalibrator(calibration)
    else:
        image_points = np.asarray(calibration_config["image_points_px"], dtype=np.float64)
        pitch_points = np.asarray(calibration_config["pitch_points_m"], dtype=np.float64)
        width = int(np.max(image_points[:, 0]) + 20)
        height = int(np.max(image_points[:, 1]) + 20)
        calibration = estimate_homography(
            image_points,
            pitch_points,
            HomographyConfig(
                image_size_px=(width, height),
                min_spatial_coverage=0.1,
                max_reprojection_error=0.01,
            ),
        )
        calibrator = StaticCalibrator(calibration)
    return OfflinePipeline(
        detector,
        tracker,
        calibrator,
        config_path=path,
        reader_factory=reader_factory,
    )

