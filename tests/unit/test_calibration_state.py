from pitchtrack.calibration.state import CalibrationStateConfig


def test_state_config_rejects_invalid_prediction_limits() -> None:
    try:
        CalibrationStateConfig(max_prediction_age_frames=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero prediction age must be rejected")


def test_state_config_requires_strict_decay() -> None:
    for value in (0.0, 1.0, 1.2):
        try:
            CalibrationStateConfig(prediction_confidence_decay=value)
        except ValueError as exc:
            assert "between" in str(exc)
        else:
            raise AssertionError(f"decay {value} must be rejected")

