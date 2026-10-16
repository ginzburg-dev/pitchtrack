import numpy as np
import pytest

from pitchtrack.field_observation.base import FieldObservation, FieldObservationProvider
from pitchtrack.field_observation.manual import ManualFieldObservationProvider


def observation() -> FieldObservation:
    return FieldObservation(
        image_points_px=np.array(
            [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]
        ),
        pitch_points_m=np.array(
            [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]]
        ),
        confidences=np.array([0.9, 0.8, 0.9, 0.8]),
        source="manual-test",
    )


def test_manual_provider_satisfies_replaceable_provider_contract() -> None:
    provider = ManualFieldObservationProvider(observation())

    assert isinstance(provider, FieldObservationProvider)
    returned = provider.observe(np.zeros((100, 100, 3), dtype=np.uint8))
    assert returned.source == "manual-test"
    np.testing.assert_allclose(returned.image_points_px, observation().image_points_px)


def test_empty_observation_has_no_correspondences() -> None:
    empty = FieldObservation.empty(source="no-lines")

    assert empty.is_empty
    assert empty.image_points_px.shape == (0, 2)
    assert empty.pitch_points_m.shape == (0, 2)


def test_observation_rejects_mismatched_correspondence_counts() -> None:
    with pytest.raises(ValueError, match="same number"):
        FieldObservation(
            image_points_px=np.zeros((4, 2)),
            pitch_points_m=np.zeros((3, 2)),
            confidences=np.ones(4),
            source="broken",
        )

