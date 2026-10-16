import json
import subprocess
import sys
from pathlib import Path


def test_virtual_ptz_benchmark_writes_versioned_reproducible_metrics(
    tmp_path: Path,
) -> None:
    transforms = tmp_path / "transforms.jsonl"
    records = []
    for index, x in enumerate((0.0, 5.0, 12.0)):
        records.append(
            {
                "frame_index": index,
                "source_frame_index": 100 + index,
                "crop_window_source_px": [x, 20.0, 1000.0, 562.5],
                "H_crop_to_source": [[1, 0, x], [0, 1, 20], [0, 0, 1]],
                "H_crop_to_pitch": None,
            }
        )
    transforms.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = tmp_path / "benchmark"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pitchtrack",
            "benchmark",
            "--suite",
            "virtual-ptz",
            "--input",
            str(transforms),
            "--out",
            str(output),
            "--dataset",
            "SoccerTrack v2",
            "--split",
            "public-demo",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == 1
    assert metrics["suite"] == "virtual-ptz"
    assert metrics["dataset"] == "SoccerTrack v2"
    assert metrics["results"]["frame_count"] == 3
    assert metrics["results"]["transform_completeness"] == 1.0
    assert metrics["results"]["pitch_homography_completeness"] is None
    assert metrics["results"]["dropped_frames"] is None
    assert metrics["environment"]["python_version"]
