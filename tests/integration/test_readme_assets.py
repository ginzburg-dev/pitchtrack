import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_readme_embeds_only_existing_manifested_assets() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\((assets/[^)]+)\)", readme)
    manifest = json.loads((ROOT / "assets/manifest.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["assets"]}

    assert {"assets/architecture.svg", "assets/benchmark.svg"}.issubset(links)
    for relative in links:
        path = ROOT / relative
        assert path.is_file(), relative
        name = path.relative_to(ROOT / "assets").as_posix()
        assert name in entries
        assert path.stat().st_size == entries[name]["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entries[name]["sha256"]


def test_readme_is_github_native_and_contains_no_unsupported_demo_claim() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert ("port" + "folio") not in readme
    assert "linear-gradient" not in readme
    assert "glassmorphism" not in readme
    assert "assets/demo-overview.gif" not in readme
    assert "official jac@5 | pending" in readme
    assert "annotation-backed reference" in readme
