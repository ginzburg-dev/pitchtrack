import json
from pathlib import Path

from pitchtrack.cli import main
from tests.unit.test_media_assets import make_run


def test_make_assets_cli_reports_manifest(tmp_path: Path, capsys) -> None:
    run = tmp_path / "run"
    output = tmp_path / "assets"
    make_run(run, source_kind="annotation-reference")

    result = main(
        [
            "make-assets",
            "--run",
            str(run),
            "--out",
            str(output),
            "--source-kind",
            "annotation-reference",
        ]
    )

    assert result == 0
    assert "manifest=" in capsys.readouterr().out
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in manifest["assets"]} == {
        "demo-overview.gif",
        "dynamic-calibration.gif",
        "failure-handling.gif",
        "full-demo.mp4",
    }
