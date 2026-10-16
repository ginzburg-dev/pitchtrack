import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_makefile_exposes_uv_quality_targets() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert any(
        dependency.startswith("ruff")
        for dependency in project["project"]["optional-dependencies"]["dev"]
    )
    assert any(
        dependency.lower().startswith("pillow")
        for dependency in project["project"]["optional-dependencies"]["dev"]
    )
    assert project["tool"]["ruff"]["target-version"] == "py311"
    for target in ("sync", "test", "lint", "format-check", "format", "quality", "build"):
        assert re.search(rf"^{target}:", makefile, re.MULTILINE)
    assert re.search(r"^quality: lint format-check test(?:\s+##.*)?$", makefile, re.MULTILINE)
    assert "uv sync --extra dev" in makefile
    assert "uv run pytest -q" in makefile
    assert "uv run ruff check ." in makefile
    assert "uv run ruff format --check ." in makefile
    assert "uv build" in makefile
