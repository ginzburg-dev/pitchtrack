from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pitchtrack.media.clips import FrameWindow, select_dynamic_window, select_failure_window
from pitchtrack.media.encode import encode_gif, encode_mp4, enforce_asset_size, video_metadata


@dataclass(frozen=True, slots=True)
class AssetManifest:
    path: Path
    source_kind: str
    assets: tuple[dict[str, Any], ...]


def build_media_assets(
    run_dir: str | Path,
    output_dir: str | Path,
    source_kind: str,
) -> AssetManifest:
    if source_kind not in {"model", "annotation-reference"}:
        raise ValueError("source_kind must be 'model' or 'annotation-reference'")
    run = Path(run_dir)
    output = Path(output_dir)
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    recorded_kind = metadata.get("source_kind")
    if recorded_kind != source_kind:
        raise ValueError(
            f"source_kind mismatch: run records {recorded_kind!r}, requested {source_kind!r}"
        )
    video = run / "annotated.mp4"
    calibration = _read_jsonl(run / "calibration.jsonl")
    tracks = _read_jsonl(run / "tracks.jsonl")
    fps, video_frames, _ = video_metadata(video)
    declared = int(metadata.get("frames_processed", -1))
    if not (video_frames == len(calibration) == len(tracks) == declared):
        raise ValueError(
            "run artifacts are not synchronized: "
            f"video={video_frames}, calibration={len(calibration)}, "
            f"tracks={len(tracks)}, metadata={declared}"
        )
    output.mkdir(parents=True, exist_ok=True)
    label = "MODEL OUTPUT" if source_kind == "model" else "ANNOTATION REFERENCE"
    windows: list[tuple[str, FrameWindow, int]] = [
        (
            "demo-overview.gif",
            FrameWindow(0, min(video_frames, max(1, int(round(fps * 10))))),
            15 * 1024 * 1024,
        ),
        (
            "dynamic-calibration.gif",
            select_dynamic_window(calibration, fps=fps),
            8 * 1024 * 1024,
        ),
    ]
    failure = select_failure_window(calibration, fps=fps)
    if failure is not None:
        windows.append(("failure-handling.gif", failure, 8 * 1024 * 1024))
    assets: list[dict[str, Any]] = []
    for name, window, limit in windows:
        destination = output / name
        encode_gif(
            video,
            destination,
            start=window.start,
            stop=window.stop,
            label=label,
        )
        enforce_asset_size(destination, max_bytes=limit)
        assets.append(_asset_record(destination, output, source_kind))
    full_video = output / "full-demo.mp4"
    encode_mp4(
        video,
        full_video,
        start=0,
        stop=min(video_frames, max(1, int(round(fps * 45)))),
        label=label,
    )
    enforce_asset_size(full_video, max_bytes=50 * 1024 * 1024)
    assets.append(_asset_record(full_video, output, source_kind))
    payload = {
        "schema_version": 1,
        "source_kind": source_kind,
        "source_run": str(run.resolve()),
        "source_metadata_sha256": _sha256(run / "run_metadata.json"),
        "assets": assets,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return AssetManifest(manifest_path, source_kind, tuple(assets))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _asset_record(path: Path, root: Path, source_kind: str) -> dict[str, Any]:
    return {
        "name": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "source_kind": source_kind,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
