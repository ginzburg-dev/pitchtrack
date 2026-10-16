from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class MissingDemoSourceError(FileNotFoundError):
    """Raised when a gated/local dataset asset is unavailable."""


@dataclass(frozen=True, slots=True)
class DemoManifest:
    dataset: str
    match_id: str
    half: int
    license_name: str
    source_url: str
    citation: str
    video_template: str
    annotations_template: str
    duration_s: float
    output_size: tuple[int, int]
    visual_gates: dict[str, float]

    def attribution(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "match_id": self.match_id,
            "half": self.half,
            "license": self.license_name,
            "source_url": self.source_url,
            "citation": self.citation,
            "virtual_camera": True,
            "disclosure": "Virtual PTZ crop derived from fixed panoramic footage.",
        }


@dataclass(frozen=True, slots=True)
class DemoAssets:
    video_path: Path
    annotations_path: Path


def load_demo_manifest(path: str | Path) -> DemoManifest:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DemoManifest(
        dataset=str(data["dataset"]),
        match_id=str(data["match_id"]),
        half=int(data["half"]),
        license_name=str(data["license"]),
        source_url=str(data["source_url"]),
        citation=str(data["citation"]),
        video_template=str(data["video_template"]),
        annotations_template=str(data["annotations_template"]),
        duration_s=float(data["duration_s"]),
        output_size=(int(data["output_width"]), int(data["output_height"])),
        visual_gates={key: float(value) for key, value in data["visual_gates"].items()},
    )


def resolve_soccertrack_assets(
    root: str | Path,
    manifest: DemoManifest,
) -> DemoAssets:
    root_path = Path(root)
    half_ordinals = {1: "1st", 2: "2nd"}
    try:
        half_ordinal = half_ordinals[manifest.half]
    except KeyError as error:
        raise ValueError("SoccerTrack half must be 1 or 2") from error
    values = {
        "match_id": manifest.match_id,
        "half": manifest.half,
        "half_ordinal": half_ordinal,
    }
    video = root_path / manifest.video_template.format(**values)
    annotations = root_path / manifest.annotations_template.format(**values)
    missing = [path for path in (video, annotations) if not path.is_file()]
    if missing:
        expected = "\n".join(f"  - {path}" for path in missing)
        raise MissingDemoSourceError(
            "SoccerTrack v2 demo assets are missing:\n"
            f"{expected}\n"
            f"Accept the dataset terms and acquire the selected match from {manifest.source_url}.\n"
            "No download was attempted by PitchTrack."
        )
    return DemoAssets(video.resolve(), annotations.resolve())
