from dataclasses import replace
from pathlib import Path

import pytest

from pitchtrack.demo.manifest import (
    MissingDemoSourceError,
    load_demo_manifest,
    resolve_soccertrack_assets,
)


MANIFEST_PATH = Path("configs/demo/soccertrack-v2-117093.yaml")


def test_manifest_contains_public_attribution_and_virtual_camera_disclosure() -> None:
    manifest = load_demo_manifest(MANIFEST_PATH)

    attribution = manifest.attribution()

    assert manifest.match_id == "117093"
    assert manifest.half == 1
    assert attribution["license"] == "CC BY 4.0"
    assert "SoccerTrack v2" in attribution["dataset"]
    assert attribution["virtual_camera"] is True
    assert "fixed panoramic" in attribution["disclosure"]


def test_missing_dataset_reports_expected_path_and_acquisition_url(
    tmp_path: Path,
) -> None:
    manifest = load_demo_manifest(MANIFEST_PATH)

    with pytest.raises(MissingDemoSourceError) as error:
        resolve_soccertrack_assets(tmp_path, manifest)

    message = str(error.value)
    assert "117093_panorama_1st_half.mp4" in message
    assert "huggingface.co/datasets/atomscott/soccertrack-v2" in message
    assert "No download was attempted" in message


def test_second_half_uses_english_ordinal_in_asset_paths(tmp_path: Path) -> None:
    manifest = load_demo_manifest(MANIFEST_PATH)
    manifest = replace(manifest, half=2)

    with pytest.raises(MissingDemoSourceError) as error:
        resolve_soccertrack_assets(tmp_path, manifest)

    message = str(error.value)
    assert "117093_panorama_2nd_half.mp4" in message
    assert "117093_2nd.json" in message

