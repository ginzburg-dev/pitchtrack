from pitchtrack.demo.selector import (
    ClipSelectionCriteria,
    FrameCandidate,
    PlayerBox,
    select_demo_interval,
)


def frame(index: int, *, good: bool) -> FrameCandidate:
    count = 8 if good else 3
    height = 100.0 if good else 35.0
    boxes = tuple(
        PlayerBox(20.0 + player * 25.0, 30.0, 32.0 + player * 25.0, 30.0 + height)
        for player in range(count)
    )
    return FrameCandidate(index, boxes, line_visible=good, lighting_score=0.9)


def test_selector_chooses_continuous_window_that_passes_visual_gates() -> None:
    frames = [frame(index, good=2 <= index < 8) for index in range(10)]
    criteria = ClipSelectionCriteria(
        fps=2.0,
        duration_s=3.0,
        min_players=6,
        min_player_height_px=80.0,
        max_player_height_px=180.0,
        min_passing_fraction=1.0,
    )

    selection = select_demo_interval(frames, criteria)

    assert selection.start_frame == 2
    assert selection.end_frame_exclusive == 8
    assert selection.passing_fraction == 1.0
    assert selection.median_player_height_px == 100.0


def test_selector_rejects_when_no_window_passes_player_scale_gate() -> None:
    frames = [frame(index, good=False) for index in range(10)]
    criteria = ClipSelectionCriteria(fps=2.0, duration_s=3.0)

    try:
        select_demo_interval(frames, criteria)
    except ValueError as exc:
        assert "visual gates" in str(exc)
    else:
        raise AssertionError("a clip with tiny players must not be selected")
