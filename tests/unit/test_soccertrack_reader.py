import json
from pathlib import Path

import pytest

from pitchtrack.demo.soccertrack import (
    annotated_frames_to_candidates,
    iter_soccertrack_frames,
)


def write_payload(path: Path) -> None:
    payload = {
        "info": {"frame_rate": 25},
        "images": [],
        "annotations": [
            {
                "id": "300000001",
                "image_id": "3000001",
                "track_id": 7,
                "supercategory": "object",
                "attributes": {"role": "player", "team": "left"},
                "bbox_image": {"x": 10, "y": 20, "w": 30, "h": 40},
                "bbox_pitch": {
                    "x_bottom_middle": -12.5,
                    "y_bottom_middle": 8.25,
                },
            },
            {
                "id": "300000002",
                "image_id": "3000001",
                "track_id": 9,
                "supercategory": "object",
                "attributes": {"role": "referee", "team": None},
                "bbox_image": {"x": 50, "y": 10, "w": 10, "h": 20},
                "bbox_pitch": {},
            },
            {
                "id": "300000003",
                "image_id": "3000002",
                "supercategory": "pitch",
            },
            {
                "id": "300000004",
                "image_id": "3000002",
                "track_id": 7,
                "supercategory": "object",
                "attributes": {"role": "player", "team": "left"},
                "bbox_image": {"x": 12, "y": 21, "w": 30, "h": 40},
                "bbox_pitch": {
                    "x_bottom_middle": -12.0,
                    "y_bottom_middle": 8.0,
                },
            },
        ],
        "categories": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reader_groups_shipped_coco_annotations_and_normalizes_geometry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gsr.json"
    write_payload(source)

    frames = list(iter_soccertrack_frames(source))

    assert [frame.frame_index for frame in frames] == [0, 1]
    assert [entity.track_id for entity in frames[0].entities] == [7, 9]
    assert frames[0].entities[0].bbox_xyxy_px == (10.0, 20.0, 40.0, 60.0)
    assert frames[0].entities[0].pitch_xy_m == (-12.5, 8.25)
    assert frames[0].entities[1].pitch_xy_m is None
    assert frames[0].entities[1].team is None
    assert frames[1].line_visible is True


def test_reader_is_lazy_and_converts_frames_to_clip_candidates(tmp_path: Path) -> None:
    source = tmp_path / "gsr.json"
    write_payload(source)

    iterator = iter_soccertrack_frames(source)
    assert iter(iterator) is iterator
    first = next(iterator)
    candidates = annotated_frames_to_candidates([first])

    assert candidates[0].frame_index == 0
    assert candidates[0].player_boxes[0].height == 40.0


def test_reader_rejects_out_of_order_frame_ids(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    payload = {
        "annotations": [
            {"image_id": "3000002", "supercategory": "pitch"},
            {"image_id": "3000001", "supercategory": "pitch"},
        ]
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="chronological"):
        list(iter_soccertrack_frames(source))
