import numpy as np

from pitchtrack.schemas import Detection
from pitchtrack.tracking.bytetrack import ByteTrackAdapter, ByteTrackConfig

FRAME = np.zeros((100, 160, 3), dtype=np.uint8)


def detection(x: float) -> Detection:
    return Detection((x, 20.0, x + 20.0, 80.0), 0.9)


def test_tracker_keeps_id_across_short_occlusion() -> None:
    tracker = ByteTrackAdapter(ByteTrackConfig(max_age_frames=2, min_iou=0.2))
    first = tracker.update([detection(10)], FRAME)
    missing = tracker.update([], FRAME)
    recovered = tracker.update([detection(12)], FRAME)

    assert first[0].track_id is not None
    assert missing == []
    assert recovered[0].track_id == first[0].track_id


def test_tracker_removes_track_after_configured_age() -> None:
    tracker = ByteTrackAdapter(ByteTrackConfig(max_age_frames=1, min_iou=0.2))
    first_id = tracker.update([detection(10)], FRAME)[0].track_id
    tracker.update([], FRAME)
    tracker.update([], FRAME)
    new_id = tracker.update([detection(11)], FRAME)[0].track_id

    assert new_id is not None
    assert new_id != first_id


def test_tracker_never_emits_missing_ids() -> None:
    tracker = ByteTrackAdapter()
    tracks = tracker.update([detection(10), detection(80)], FRAME)

    assert all(track.track_id is not None for track in tracks)
    assert len({track.track_id for track in tracks}) == 2
