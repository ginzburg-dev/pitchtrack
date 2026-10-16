# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-16

### Added

- Offline pipeline for a single continuous football-camera shot, with annotated MP4 and JSONL outputs.
- Dynamic planar field homography estimation using field observations, masked optical flow, RANSAC, temporal validation, and reinitialisation.
- Player detection and multi-object tracking interfaces, with optional Ultralytics and ByteTrack adapters.
- Image-space tracking retained when pitch geometry is unreliable; no fabricated pitch coordinates are emitted.
- Evaluation plumbing for SoccerNet calibration, SoccerTrack v2 tracking, temporal stability, and runtime measurements.
- Reproducible demo assets with provenance and licensing documentation.
- `uv`-based development workflow, quality checks, GitHub Actions CI, and tagged GitHub Releases containing Python distributions.
- MIT license for the PitchTrack source code.

[Unreleased]: https://github.com/ginzburg-dev/pitchtrack/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ginzburg-dev/pitchtrack/releases/tag/v0.1.0
