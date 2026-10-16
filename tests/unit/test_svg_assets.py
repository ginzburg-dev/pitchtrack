import json
import xml.etree.ElementTree as ET
from pathlib import Path

from pitchtrack.media.svg import render_architecture_svg, render_benchmark_svg


def test_architecture_svg_is_light_strict_and_names_real_components(tmp_path: Path) -> None:
    output = tmp_path / "architecture.svg"

    render_architecture_svg(output)

    text = output.read_text(encoding="utf-8")
    ET.parse(output)
    assert "Player detector" in text
    assert "Temporal calibration" in text
    assert "Pitch projection" in text
    assert "#ffffff" in text
    assert "linearGradient" not in text
    assert "filter=" not in text


def test_benchmark_svg_uses_verified_values_and_pending_for_missing_scores(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "dataset": "fixture <not benchmark>",
                "frames_processed": 12,
                "throughput_fps": 42.25,
                "calibration_status_counts": {"valid": 8, "predicted": 3, "invalid": 1},
                "dropped_frames": 0,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "benchmark.svg"

    render_benchmark_svg(metrics, output)

    text = output.read_text(encoding="utf-8")
    ET.parse(output)
    assert "12" in text
    assert "42.25" in text
    assert "fixture &lt;not benchmark&gt;" in text
    assert text.count("pending") >= 3
    assert "Input SHA-256" in text
    assert "linearGradient" not in text
