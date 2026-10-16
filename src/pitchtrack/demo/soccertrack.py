from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson

from pitchtrack.demo.selector import FrameCandidate, PlayerBox


@dataclass(frozen=True, slots=True)
class AnnotatedEntity:
    track_id: int
    bbox_xyxy_px: tuple[float, float, float, float]
    pitch_xy_m: tuple[float, float] | None
    role: str | None
    team: str | None


@dataclass(frozen=True, slots=True)
class AnnotatedFrame:
    frame_index: int
    entities: tuple[AnnotatedEntity, ...]
    line_visible: bool


def iter_soccertrack_frames(path: str | Path) -> Iterator[AnnotatedFrame]:
    """Stream the as-shipped SoccerNet-COCO annotations one frame at a time."""
    with Path(path).open("rb") as handle:
        annotations = ijson.items(handle, "annotations.item")
        current_index: int | None = None
        entities: list[AnnotatedEntity] = []
        line_visible = False
        for annotation in annotations:
            frame_index = _frame_index(annotation["image_id"])
            if current_index is not None and frame_index < current_index:
                raise ValueError("SoccerTrack annotations must be chronological")
            if current_index is not None and frame_index != current_index:
                yield AnnotatedFrame(current_index, tuple(entities), line_visible)
                entities = []
                line_visible = False
            current_index = frame_index
            category = annotation.get("supercategory")
            if category == "pitch":
                line_visible = True
            elif category == "object":
                entity = _entity(annotation)
                if entity is not None:
                    entities.append(entity)
        if current_index is not None:
            yield AnnotatedFrame(current_index, tuple(entities), line_visible)


def annotated_frames_to_candidates(
    frames: Iterable[AnnotatedFrame],
) -> list[FrameCandidate]:
    return [
        FrameCandidate(
            frame_index=frame.frame_index,
            player_boxes=tuple(PlayerBox(*entity.bbox_xyxy_px) for entity in frame.entities),
            line_visible=frame.line_visible,
        )
        for frame in frames
    ]


def _frame_index(value: Any) -> int:
    if isinstance(value, str):
        if len(value) < 6 or not value[-6:].isdigit():
            raise ValueError(f"unsupported SoccerTrack image_id: {value!r}")
        return int(value[-6:]) - 1
    return int(value)


def _entity(annotation: dict[str, Any]) -> AnnotatedEntity | None:
    bbox = annotation.get("bbox_image")
    track_id = annotation.get("track_id")
    if not isinstance(bbox, dict) or track_id is None:
        return None
    x = float(bbox["x"])
    y = float(bbox["y"])
    width = float(bbox["w"])
    height = float(bbox["h"])
    if width <= 0 or height <= 0:
        return None
    pitch = annotation.get("bbox_pitch") or {}
    pitch_x = pitch.get("x_bottom_middle")
    pitch_y = pitch.get("y_bottom_middle")
    pitch_xy = (
        (float(pitch_x), float(pitch_y))
        if pitch_x is not None and pitch_y is not None
        else None
    )
    attributes = annotation.get("attributes") or {}
    return AnnotatedEntity(
        track_id=int(track_id),
        bbox_xyxy_px=(x, y, x + width, y + height),
        pitch_xy_m=pitch_xy,
        role=attributes.get("role"),
        team=attributes.get("team"),
    )
