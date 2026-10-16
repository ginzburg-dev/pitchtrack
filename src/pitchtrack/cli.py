from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from pitchtrack.demo.manifest import (
    MissingDemoSourceError,
    load_demo_manifest,
    resolve_soccertrack_assets,
)
from pitchtrack.demo.virtual_ptz import generate_demo_clip
from pitchtrack.evaluation.benchmark import run_benchmark
from pitchtrack.media import build_media_assets
from pitchtrack.pipeline import build_pipeline
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

    demo_parser = subparsers.add_parser(
        "make-demo-clip",
        help="Create a licensed virtual-PTZ demo clip from local SoccerTrack v2 data.",
    )
    demo_parser.add_argument("--source-root", required=True, type=Path)
    demo_parser.add_argument("--out", required=True, type=Path)
    demo_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/demo/soccertrack-v2-117093.yaml"),
    )
    demo_parser.add_argument("--match-id", default="117093")
    demo_parser.add_argument("--half", type=int, default=1, choices=(1, 2))

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Evaluate generated artifacts without fabricating missing metrics."
    )
    benchmark_parser.add_argument(
        "--suite",
        required=True,
        choices=("virtual-ptz", "soccernet-calibration", "tracking"),
    )
    benchmark_parser.add_argument("--input", required=True, type=Path)
    benchmark_parser.add_argument("--out", required=True, type=Path)
    benchmark_parser.add_argument("--dataset")
    benchmark_parser.add_argument("--split")

    assets_parser = subparsers.add_parser(
        "make-assets", help="Build verified GitHub media from a synchronized run."
    )
    assets_parser.add_argument("--run", required=True, type=Path)
    assets_parser.add_argument("--out", required=True, type=Path)
    assets_parser.add_argument(
        "--source-kind",
        required=True,
        choices=("model", "annotation-reference"),
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
        try:
            pipeline = build_pipeline(
                args.config,
                calibration_mode=args.calibration_mode,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        summary = pipeline.run(args.video, args.out)
        print(f"frames_processed={summary.frames_processed}")
        print(f"output={summary.output_dir}")
        return 0
    if args.command == "make-demo-clip":
        manifest = replace(
            load_demo_manifest(args.manifest),
            match_id=str(args.match_id),
            half=int(args.half),
        )
        try:
            assets = resolve_soccertrack_assets(args.source_root, manifest)
        except MissingDemoSourceError as exc:
            parser.error(str(exc))
        result = generate_demo_clip(assets, manifest, args.out)
        print(f"video={result.video_path}")
        print(f"transforms={result.transforms_path}")
        print(f"quality={result.quality_report_path}")
        print(f"attribution={result.attribution_path}")
        return 0
    if args.command == "benchmark":
        try:
            metrics_path = run_benchmark(
                args.suite,
                args.input,
                args.out,
                dataset=args.dataset,
                split=args.split,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(f"metrics={metrics_path}")
        return 0
    if args.command == "make-assets":
        try:
            manifest = build_media_assets(args.run, args.out, args.source_kind)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        print(f"manifest={manifest.path}")
        return 0
    parser.print_help()
    return 0
