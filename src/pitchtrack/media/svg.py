from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

BG = "#ffffff"
PANEL = "#f6f8fa"
BORDER = "#d0d7de"
TEXT = "#24292f"
MUTED = "#57606a"
ACCENT = "#0969da"


def render_architecture_svg(path: str | Path) -> Path:
    nodes = [
        (40, 90, 160, 64, "Video frame", "decoded BGR"),
        (250, 40, 190, 64, "Player detector", "pitchtrack.detection"),
        (250, 140, 190, 64, "Field observer", "semantic landmarks"),
        (490, 40, 190, 64, "Image-space tracker", "persistent player IDs"),
        (490, 140, 190, 64, "Optical flow + RANSAC", "moving objects masked"),
        (730, 140, 190, 64, "Temporal calibration", "H(t) + confidence"),
        (730, 40, 190, 64, "Pitch projection", "raw + smoothed tracks"),
        (970, 90, 160, 64, "Artifacts", "MP4 + JSONL + metrics"),
    ]
    arrows = [
        (200, 122, 250, 72),
        (200, 122, 250, 172),
        (440, 72, 490, 72),
        (440, 172, 490, 172),
        (680, 172, 730, 172),
        (825, 140, 825, 104),
        (680, 72, 730, 72),
        (920, 72, 970, 122),
    ]
    body = [
        '<rect width="1170" height="250" fill="#ffffff"/>',
        _text(40, 32, "PitchTrack offline pipeline", 20, weight=600),
        _text(40, 57, "Implemented modules and artifact flow", 13, color=MUTED),
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#57606a"/></marker></defs>',
    ]
    body.extend(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.5" marker-end="url(#arrow)"/>'
        for x1, y1, x2, y2 in arrows
    )
    for x, y, width, height, title, subtitle in nodes:
        body.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{PANEL}" stroke="{BORDER}"/>'
        )
        body.append(_text(x + 12, y + 26, title, 14, weight=600))
        body.append(_text(x + 12, y + 48, subtitle, 11, color=MUTED))
    return _write_svg(path, 1170, 250, body, "Implemented PitchTrack architecture")


def render_benchmark_svg(metrics_path: str | Path, path: str | Path) -> Path:
    source = Path(metrics_path)
    metrics = json.loads(source.read_text(encoding="utf-8"))
    dataset = str(metrics.get("dataset", "Local synthetic fixture smoke check"))
    status = metrics.get("calibration_status_counts", {})
    rows = [
        ("Processed frames", str(metrics.get("frames_processed", "pending"))),
        ("Offline throughput", _number(metrics.get("throughput_fps"), " FPS")),
        (
            "Calibration status",
            f"{status.get('valid', 0)} valid / {status.get('predicted', 0)} predicted / {status.get('invalid', 0)} invalid",
        ),
        ("Dropped frames", str(metrics.get("dropped_frames", "pending"))),
        ("Official JaC@5", "pending"),
        ("Official completeness", "pending"),
        ("Official HOTA", "pending"),
    ]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    body = [
        '<rect width="900" height="430" fill="#ffffff"/>',
        _text(36, 38, "Verified results", 20, weight=600),
        _text(36, 64, dataset, 13, color=MUTED),
        '<line x1="36" y1="82" x2="864" y2="82" stroke="#d0d7de"/>',
    ]
    y = 116
    for label, value in rows:
        body.append(_text(48, y, label, 13, color=MUTED))
        body.append(
            _text(520, y, value, 13, weight=600, color=ACCENT if value == "pending" else TEXT)
        )
        body.append(f'<line x1="36" y1="{y + 17}" x2="864" y2="{y + 17}" stroke="{BORDER}"/>')
        y += 42
    body.append(_text(36, 414, f"Input SHA-256: {digest}", 10, color=MUTED))
    return _write_svg(path, 900, 430, body, "Metrics generated from a hashed report")


def _number(value: Any, suffix: str) -> str:
    return "pending" if value is None else f"{float(value):.2f}{suffix}"


def _text(
    x: float,
    y: float,
    value: str,
    size: int,
    *,
    color: str = TEXT,
    weight: int = 400,
) -> str:
    escaped = html.escape(str(value), quote=True)
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" '
        f'font-size="{size}" font-weight="{weight}">{escaped}</text>'
    )


def _write_svg(
    path: str | Path,
    width: int,
    height: int,
    body: list[str],
    metadata: str,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f"<metadata>{html.escape(metadata)}</metadata>" + "".join(body) + "</svg>"
    )
    destination.write_text(content, encoding="utf-8")
    return destination
