import json
from pathlib import Path

import cv2
import numpy as np

from pitchtrack.calibration.geometry import project_points
from pitchtrack.demo.selector import PlayerBox
from pitchtrack.demo.virtual_ptz import (
    CropWindow,
    compose_crop_to_pitch,
    compute_target_window,
    crop_to_source_homography,
    generate_demo_clip,
    render_virtual_ptz_video,
    smooth_crop_windows,
)
from pitchtrack.demo.manifest import DemoAssets, DemoManifest


def test_crop_transform_maps_output_corners_to_source_window() -> None:
    window = CropWindow(x=100.0, y=50.0, width=800.0, height=450.0)

    transform = crop_to_source_homography(window, output_size=(1920, 1080))
    mapped = project_points(
        np.array([[0.0, 0.0], [1920.0, 1080.0]]),
        transform,
    )

    np.testing.assert_allclose(mapped, [[100.0, 50.0], [900.0, 500.0]])


def test_crop_to_pitch_composition_matches_two_step_projection() -> None:
    window = CropWindow(x=100.0, y=50.0, width=800.0, height=450.0)
    crop_to_source = crop_to_source_homography(window, (1920, 1080))
    source_to_pitch = np.array(
        [[0.1, 0.0, -2.0], [0.0, 0.1, 3.0], [0.0, 0.0, 1.0]]
    )
    points = np.array([[100.0, 200.0], [1000.0, 500.0]])

    composed = compose_crop_to_pitch(source_to_pitch, crop_to_source)

    np.testing.assert_allclose(
        project_points(points, composed),
        project_points(project_points(points, crop_to_source), source_to_pitch),
    )


def test_smoothing_limits_pan_and_zoom_and_keeps_crop_in_bounds() -> None:
    targets = [
        CropWindow(0.0, 0.0, 800.0, 450.0),
        CropWindow(1200.0, 500.0, 400.0, 225.0),
        CropWindow(1300.0, 550.0, 400.0, 225.0),
    ]

    smoothed = smooth_crop_windows(
        targets,
        source_size=(2048, 1152),
        output_size=(1920, 1080),
        max_pan_px_per_frame=40.0,
        max_zoom_fraction_per_frame=0.05,
    )

    centers = np.array([[item.center_x, item.center_y] for item in smoothed])
    widths = np.array([item.width for item in smoothed])
    assert np.max(np.linalg.norm(np.diff(centers, axis=0), axis=1)) <= 40.0 + 1e-6
    assert np.max(np.abs(np.diff(widths)) / widths[:-1]) <= 0.05 + 1e-6
    assert all(item.x >= 0 and item.y >= 0 for item in smoothed)
    assert all(item.x + item.width <= 2048.0 + 1e-6 for item in smoothed)
    assert all(item.y + item.height <= 1152.0 + 1e-6 for item in smoothed)


def test_target_window_frames_players_at_16_by_9() -> None:
    boxes = (
        PlayerBox(400.0, 300.0, 460.0, 500.0),
        PlayerBox(900.0, 250.0, 960.0, 470.0),
    )

    target = compute_target_window(
        boxes,
        source_size=(2048, 1152),
        output_size=(1920, 1080),
    )

    assert target.width / target.height == 16 / 9
    assert target.x <= 400.0
    assert target.x + target.width >= 960.0


def test_renderer_writes_output_frames_and_aligned_transform_jsonl(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180)
    )
    assert writer.isOpened()
    for value in (30, 80, 130):
        writer.write(np.full((180, 320, 3), value, dtype=np.uint8))
    writer.release()
    output = tmp_path / "virtual.mp4"
    transforms = tmp_path / "virtual.transforms.jsonl"
    windows = [CropWindow(0.0, 0.0, 320.0, 180.0)] * 3

    render_virtual_ptz_video(
        source,
        output,
        transforms,
        windows,
        output_size=(160, 90),
        H_source_to_pitch=np.eye(3),
    )

    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 90
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()
    records = [json.loads(line) for line in transforms.read_text().splitlines()]
    assert [record["frame_index"] for record in records] == [0, 1, 2]
    assert all(record["H_crop_to_pitch"] is not None for record in records)


def test_generate_demo_clip_writes_quality_and_attribution_reports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180)
    )
    assert writer.isOpened()
    annotations = []
    for index in range(4):
        writer.write(np.full((180, 320, 3), 40 + index * 20, dtype=np.uint8))
        annotations.append(
            {
                "frame_index": index,
                "line_visible": True,
                "players": [
                    {"bbox_xyxy_px": [40, 40, 70, 120]},
                    {"bbox_xyxy_px": [180, 35, 210, 125]},
                ],
            }
        )
    writer.release()
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({"frames": annotations}), encoding="utf-8")
    manifest = DemoManifest(
        dataset="SoccerTrack v2 test fixture",
        match_id="117093",
        half=1,
        license_name="CC BY 4.0",
        source_url="https://example.invalid/dataset",
        citation="fixture citation",
        video_template="unused",
        annotations_template="unused",
        duration_s=0.3,
        output_size=(160, 90),
        visual_gates={
            "min_players": 2,
            "min_player_height_px": 50,
            "max_player_height_px": 120,
            "min_passing_fraction": 1.0,
            "min_lighting_score": 0.5,
        },
    )
    output_dir = tmp_path / "demo"

    result = generate_demo_clip(
        DemoAssets(source, annotation_path),
        manifest,
        output_dir,
    )

    assert result.video_path.is_file()
    assert result.transforms_path.is_file()
    quality = json.loads(result.quality_report_path.read_text("utf-8"))
    attribution = json.loads(result.attribution_path.read_text("utf-8"))
    assert quality["selected_frame_count"] == 3
    assert quality["passing_fraction"] == 1.0
    assert attribution["license"] == "CC BY 4.0"
    assert attribution["virtual_camera"] is True
