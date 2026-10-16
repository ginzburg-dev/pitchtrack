from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pitchtrack import __version__


def run_benchmark(
    suite: str,
    input_path: str | Path,
    output_dir: str | Path,
    *,
    dataset: str | None = None,
    split: str | None = None,
) -> Path:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Benchmark input not found: {source}")
    if suite == "virtual-ptz":
        results = _virtual_ptz_metrics(source)
    elif suite in {"soccernet-calibration", "tracking"}:
        results = _external_adapter_manifest(source, suite)
    else:
        raise ValueError(f"unsupported benchmark suite: {suite}")
    payload = {
        "schema_version": 1,
        "suite": suite,
        "dataset": dataset,
        "split": split,
        "input": {
            "path": str(source.resolve()),
            "sha256": _sha256(source),
        },
        "environment": {
            "pitchtrack_version": __version__,
            "python_version": platform.python_version(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "results": results,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "metrics.json"
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return destination


def _virtual_ptz_metrics(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("virtual-PTZ transform JSONL is empty")
    indexes = [int(record["frame_index"]) for record in records]
    if indexes != list(range(len(records))):
        raise ValueError("virtual-PTZ frame indexes must be contiguous from zero")
    valid_transforms = 0
    centers: list[tuple[float, float]] = []
    widths: list[float] = []
    pitch_presence: list[bool] = []
    for record in records:
        matrix = np.asarray(record.get("H_crop_to_source"), dtype=np.float64)
        valid_transforms += int(matrix.shape == (3, 3) and np.all(np.isfinite(matrix)))
        x, y, width, height = (
            float(value) for value in record["crop_window_source_px"]
        )
        centers.append((x + width / 2.0, y + height / 2.0))
        widths.append(width)
        pitch_presence.append(record.get("H_crop_to_pitch") is not None)
    pan_steps = [
        float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
        for left, right in zip(centers, centers[1:])
    ]
    zoom_steps = [
        abs(right - left) / left
        for left, right in zip(widths, widths[1:])
        if left > 0
    ]
    return {
        "frame_count": len(records),
        "transform_completeness": valid_transforms / len(records),
        "pitch_homography_completeness": (
            sum(pitch_presence) / len(records) if any(pitch_presence) else None
        ),
        "max_pan_px_per_frame": max(pan_steps) if pan_steps else None,
        "max_zoom_fraction_per_frame": max(zoom_steps) if zoom_steps else None,
        "dropped_frames": None,
    }


def _external_adapter_manifest(path: Path, suite: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "adapter_input_schema": payload.get("schema_version"),
        "record_count": len(payload.get("records", [])),
        "official_metric": None,
        "note": (
            f"{suite} input validated; run the official dataset evaluator to populate "
            "the official metric. PitchTrack does not fabricate an unavailable score."
        ),
        "dropped_frames": None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
