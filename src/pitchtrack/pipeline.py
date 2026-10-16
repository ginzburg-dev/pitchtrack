from __future__ import annotations

import hashlib
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
from pitchtrack.field_observation.base import FieldObservation, FieldObservationProvider
from pitchtrack.evaluation.performance import (
    dropped_frames as count_dropped_frames,
    summarize_stage_latencies,
)
from pitchtrack.projection.projector import PitchProjector
from pitchtrack.projection.smoothing import PitchTrackSmoother
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


def moving_object_mask(
    frame_shape: tuple[int, int], detections: Sequence[Detection]
) -> np.ndarray:
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy_px)
        x1, x2 = max(0, x1), min(width, x2 + 1)
        y1, y2 = max(0, y1), min(height, y2 + 1)
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 255
    return mask


class FieldCalibrationSource:
    """Turns semantic field observations into per-frame calibration results."""

    def __init__(
        self,
        mode: str,
        observer: FieldObservationProvider,
        homography: HomographyConfig,
        *,
        state: CalibrationStateConfig = CalibrationStateConfig(),
        flow: OpticalFlowConfig = OpticalFlowConfig(),
    ) -> None:
        if mode not in {"baseline_deeplab", "baseline_tvcalib", "hybrid_temporal"}:
            raise ValueError(f"unsupported calibration mode: {mode}")
        self.mode = mode
        self.observer = observer
        self.hybrid = (
            HybridTemporalCalibrator(
                HybridCalibrationConfig(
                    temporal=TemporalCalibrationConfig(homography=homography, state=state),
                    flow=flow,
                )
            )
            if mode == "hybrid_temporal"
            else None
        )
        self.homography = homography

    def update(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        detections: Sequence[Detection] = (),
    ) -> CalibrationResult:
        if self.hybrid is None:
            observation = self.observer.observe(frame_bgr)
            if observation.is_empty:
                return CalibrationResult.invalid("field_observation_empty")
            return estimate_homography(
                observation.image_points_px,
                observation.pitch_points_m,
                self.homography,
            )
        needs_observation = self.hybrid.temporal.should_request_observation(frame_index)
        observation = self.observer.observe(frame_bgr) if needs_observation else None
        mask = moving_object_mask(frame_bgr.shape[:2], detections)
        return self.hybrid.update(
            frame_bgr,
            mask,
            observation,
            frame_index=frame_index,
        )


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
        source_kind: str = "model",
        reader_factory: Callable[[Path], VideoReader] = VideoReader,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.calibrator = calibrator
        self.projector = PitchProjector()
        self.smoother = PitchTrackSmoother()
        self.config_path = config_path
        self.source_kind = source_kind
        self.reader_factory = reader_factory

    def run(self, video_path: str | Path, output_dir: str | Path) -> RunSummary:
        started = time.perf_counter()
        self.smoother.reset()
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        status_counts: Counter[str] = Counter()
        stage_seconds: dict[str, list[float]] = {
            stage: []
            for stage in ("detect", "track", "calibrate", "project", "render", "write")
        }
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
                    stage_started = time.perf_counter()
                    detections = tuple(self.detector.detect(frame.image_bgr))
                    stage_seconds["detect"].append(time.perf_counter() - stage_started)
                    stage_started = time.perf_counter()
                    image_tracks = tuple(
                        self.tracker.update(detections, frame.image_bgr)
                    )
                    stage_seconds["track"].append(time.perf_counter() - stage_started)
                    stage_started = time.perf_counter()
                    calibration = self.calibrator.update(
                        frame.image_bgr, frame.frame_index, detections
                    )
                    stage_seconds["calibrate"].append(
                        time.perf_counter() - stage_started
                    )
                    stage_started = time.perf_counter()
                    raw_tracks = tuple(
                        self.projector.project(track, calibration)
                        for track in image_tracks
                    )
                    if calibration.status is CalibrationStatus.INVALID:
                        self.smoother.reset()
                    projected_tracks = tuple(
                        replace(
                            track,
                            smoothed_field_xy_m=self.smoother.update(
                                track.track_id,
                                track.field_xy_m,
                                calibration.status,
                            ),
                        )
                        for track in raw_tracks
                    )
                    stage_seconds["project"].append(time.perf_counter() - stage_started)
                    result = FrameResult(
                        frame_index=frame.frame_index,
                        timestamp_s=frame.timestamp_s,
                        tracks=projected_tracks,
                        calibration=calibration,
                    )
                    stage_started = time.perf_counter()
                    rendered = render_annotated_frame(
                        frame.image_bgr, projected_tracks, calibration
                    )
                    stage_seconds["render"].append(time.perf_counter() - stage_started)
                    stage_started = time.perf_counter()
                    writer.write(result, rendered)
                    stage_seconds["write"].append(time.perf_counter() - stage_started)
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
                    "source_kind": self.source_kind,
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
                    "stage_latency_ms": summarize_stage_latencies(stage_seconds),
                    "dropped_frames": count_dropped_frames(
                        expected_frames=(
                            reader.metadata.declared_frame_count
                            if reader.metadata.declared_frame_count > 0
                            else None
                        ),
                        processed_frames=frames_processed,
                    ),
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
        source_kind="fixture",
        reader_factory=reader_factory,
    )


def build_model_pipeline(
    config_path: str | Path,
    *,
    calibration_mode: str,
    reader_factory: Callable[[Path], VideoReader] = VideoReader,
) -> OfflinePipeline:
    """Build a local-only production pipeline from explicit model paths."""
    from pitchtrack.detection.ultralytics import UltralyticsPersonDetector
    from pitchtrack.field_observation.deeplab import DeepLabFieldObserver
    from pitchtrack.field_observation.tvcalib import TVCalibFieldObserver
    from pitchtrack.tracking.bytetrack import ByteTrackAdapter, ByteTrackConfig

    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    def local_path(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path.parent / candidate

    detector_config = config["detector"]
    detector = UltralyticsPersonDetector(
        local_path(detector_config["weights"]),
        confidence=float(detector_config.get("confidence", 0.25)),
        device=detector_config.get("device"),
    )
    tracker_config = config.get("tracker", {})
    tracker = ByteTrackAdapter(
        ByteTrackConfig(
            max_age_frames=int(tracker_config.get("max_age_frames", 30)),
            min_iou=float(tracker_config.get("min_iou", 0.2)),
        )
    )
    calibration_config = config["calibration"]
    observer_mode = (
        calibration_config.get("hybrid_observer", "baseline_deeplab")
        if calibration_mode == "hybrid_temporal"
        else calibration_mode
    )
    if observer_mode == "baseline_deeplab":
        observer = DeepLabFieldObserver(
            local_path(calibration_config["deeplab_weights"])
        )
    elif observer_mode == "baseline_tvcalib":
        observer = TVCalibFieldObserver(
            local_path(calibration_config["tvcalib_weights"])
        )
    else:
        raise ValueError(f"unsupported field observer: {observer_mode}")
    width, height = (int(value) for value in config["video"]["image_size_px"])
    homography = HomographyConfig(
        image_size_px=(width, height),
        ransac_reprojection_threshold=float(
            calibration_config.get("ransac_reprojection_threshold", 3.0)
        ),
        min_inlier_ratio=float(calibration_config.get("min_inlier_ratio", 0.5)),
        min_spatial_coverage=float(
            calibration_config.get("min_spatial_coverage", 0.01)
        ),
        max_reprojection_error=float(
            calibration_config.get("max_reprojection_error", 5.0)
        ),
    )
    source = FieldCalibrationSource(
        calibration_mode,
        observer,
        homography,
        state=CalibrationStateConfig(
            max_prediction_age_frames=int(
                calibration_config.get("max_prediction_age_frames", 15)
            ),
            prediction_confidence_decay=float(
                calibration_config.get("prediction_confidence_decay", 0.9)
            ),
            reinitialize_interval_frames=int(
                calibration_config.get("reinitialize_interval_frames", 30)
            ),
            max_anchor_disagreement=float(
                calibration_config.get("max_anchor_disagreement", 5.0)
            ),
        ),
    )
    return OfflinePipeline(
        detector,
        tracker,
        source,
        config_path=path,
        source_kind="model",
        reader_factory=reader_factory,
    )


def build_pipeline(
    config_path: str | Path,
    *,
    calibration_mode: str,
    reader_factory: Callable[[Path], VideoReader] = VideoReader,
) -> OfflinePipeline:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("fixture") is True:
        return build_fixture_pipeline(
            path,
            calibration_mode=calibration_mode,
            reader_factory=reader_factory,
        )
    return build_model_pipeline(
        path,
        calibration_mode=calibration_mode,
        reader_factory=reader_factory,
    )

