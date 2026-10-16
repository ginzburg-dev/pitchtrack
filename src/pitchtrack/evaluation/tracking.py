from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def to_hota_inputs(frames: Sequence[Sequence[dict[str, Any]]]) -> dict[str, list]:
    ids: list[list[int]] = []
    detections: list[list[list[float]]] = []
    for frame in frames:
        frame_ids: list[int] = []
        frame_boxes: list[list[float]] = []
        for track in frame:
            if track.get("track_id") is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in track["bbox_xyxy_px"])
            frame_ids.append(int(track["track_id"]))
            frame_boxes.append([x1, y1, x2 - x1, y2 - y1])
        ids.append(frame_ids)
        detections.append(frame_boxes)
    return {"ids": ids, "detections": detections}
