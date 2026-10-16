# Evaluation protocol

PitchTrack keeps presentation footage, model quality, temporal geometry, tracking,
and runtime measurements separate. A visually attractive overlay is not treated as
evidence of accuracy.

## Calibration

- **SoccerNet Calibration:** report official JaC@5 and completeness from the official
  evaluator. The local mask-level `tolerant_jaccard` helper is intended for unit and
  regression tests; it must not be relabelled as an official SoccerNet score.
- **Reprojection error:** always includes its unit (`px` or `m`). Mixed units are
  rejected.
- **Coverage:** valid, predicted, and invalid frame fractions are reported separately.
- **Recovery:** frames from the first invalid result until a new valid observation.
  Unresolved failures are counted rather than discarded.

## Temporal geometry and pitch tracks

Compare raw and smoothed pitch coordinates. Jitter is mean second-difference magnitude;
response lag is the difference between raw and smoothed threshold-crossing frames.
Smoothing state is reset whenever calibration is invalid, so stale geometry cannot
silently leak across a failure.

## Tracking

`to_hota_inputs` converts per-frame IDs and image-space boxes to the `xywh` structure
expected by a TrackEval adapter. Official HOTA is intentionally left `null` until the
official evaluator and labelled split have actually run.

## Performance

Offline runs record mean, p50, and p95 latency for detection, tracking, calibration,
projection, rendering, and artifact writing, plus end-to-end throughput. Dropped frames
are reported only if the container exposes an expected frame count; otherwise the value
is `null`, not zero.

Every benchmark result includes schema version, input SHA-256, dataset/split labels,
PitchTrack version, Python, NumPy, OpenCV, and platform metadata.

## Reproducible commands

```powershell
python -m pitchtrack benchmark `
  --suite virtual-ptz `
  --input outputs/demo/pitchtrack-demo-virtual-ptz.transforms.jsonl `
  --out outputs/benchmarks/virtual-ptz `
  --dataset "SoccerTrack v2" `
  --split public-demo
```

For `soccernet-calibration` and `tracking`, pass a JSON manifest produced alongside the
official evaluator. PitchTrack validates and hashes it but leaves the official score
empty until the upstream evaluation is run.
