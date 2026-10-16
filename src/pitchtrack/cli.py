from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitchtrack",
        description="Offline football video analytics with dynamic pitch calibration.",
    )
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0

