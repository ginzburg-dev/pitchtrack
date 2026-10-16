from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pitchtrack.pipeline import build_fixture_pipeline
from pitchtrack.schemas import CalibrationStatus
from pitchtrack.testing.fixture import create_demo_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitchtrack",
        description="Offline football video analytics with dynamic pitch calibration.",
    )
    subparsers = parser.add_subparsers(dest="command")

    fixture_parser = subparsers.add_parser(
        "demo-fixture", help="Create a deterministic synthetic geometry fixture."
    )
    fixture_parser.add_argument("--out", required=True, type=Path)
    fixture_parser.add_argument("--camera-pan-px-per-frame", type=float, default=0.0)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze one continuous video shot offline."
    )
    analyze_parser.add_argument("--video", required=True, type=Path)
    analyze_parser.add_argument("--out", required=True, type=Path)
    analyze_parser.add_argument("--config", required=True, type=Path)
    analyze_parser.add_argument(
        "--calibration-mode",
        default="fixture",
        choices=("fixture", "hybrid_temporal", "baseline_deeplab", "baseline_tvcalib"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo-fixture":
        paths = create_demo_fixture(
            args.out,
            calibration_status=CalibrationStatus.VALID,
            camera_pan_px_per_frame=args.camera_pan_px_per_frame,
        )
        print(f"video={paths.video_path}")
        print(f"config={paths.config_path}")
        return 0
    if args.command == "analyze":
        if args.calibration_mode not in {"fixture", "hybrid_temporal"}:
            parser.error(
                f"calibration mode {args.calibration_mode!r} is not available yet; "
                "use 'fixture' in the current milestone"
            )
        summary = build_fixture_pipeline(
            args.config,
            calibration_mode=args.calibration_mode,
        ).run(args.video, args.out)
        print(f"frames_processed={summary.frames_processed}")
        print(f"output={summary.output_dir}")
        return 0
    parser.print_help()
    return 0

