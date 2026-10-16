from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True, slots=True)
class PlayerBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    frame_index: int
    player_boxes: tuple[PlayerBox, ...]
    line_visible: bool
    lighting_score: float = 1.0


@dataclass(frozen=True, slots=True)
class ClipSelectionCriteria:
    fps: float
    duration_s: float = 45.0
    min_players: int = 6
    min_player_height_px: float = 80.0
    max_player_height_px: float = 180.0
    min_passing_fraction: float = 0.9
    min_lighting_score: float = 0.5


@dataclass(frozen=True, slots=True)
class ClipSelection:
    start_frame: int
    end_frame_exclusive: int
    passing_fraction: float
    median_player_height_px: float


def _passes(frame: FrameCandidate, criteria: ClipSelectionCriteria) -> bool:
    if len(frame.player_boxes) < criteria.min_players or not frame.line_visible:
        return False
    if frame.lighting_score < criteria.min_lighting_score:
        return False
    heights = [box.height for box in frame.player_boxes if box.height > 0]
    if not heights:
        return False
    median_height = float(median(heights))
    return criteria.min_player_height_px <= median_height <= criteria.max_player_height_px


def select_demo_interval(
    frames: list[FrameCandidate],
    criteria: ClipSelectionCriteria,
) -> ClipSelection:
    window_size = int(round(criteria.duration_s * criteria.fps))
    if window_size <= 0 or len(frames) < window_size:
        raise ValueError("not enough annotated frames for the requested demo duration")
    best: ClipSelection | None = None
    best_score = -1.0
    for start in range(len(frames) - window_size + 1):
        window = frames[start : start + window_size]
        passing = sum(_passes(frame, criteria) for frame in window) / window_size
        heights = [
            box.height
            for frame in window
            for box in frame.player_boxes
            if box.height > 0
        ]
        height = float(median(heights)) if heights else 0.0
        score = passing + min(len(window[0].player_boxes), 22) / 1000.0
        if passing >= criteria.min_passing_fraction and score > best_score:
            best = ClipSelection(
                start_frame=frames[start].frame_index,
                end_frame_exclusive=frames[start + window_size - 1].frame_index + 1,
                passing_fraction=passing,
                median_player_height_px=height,
            )
            best_score = score
    if best is None:
        raise ValueError("no continuous interval passes the configured visual gates")
    return best

