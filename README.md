# PitchTrack

PitchTrack is an offline football-video analytics prototype for one continuous moving
camera shot. It keeps player tracks in image space, estimates a time-varying planar
image-to-pitch homography, and emits pitch coordinates only while geometry is defensible.

The MVP deliberately excludes broadcast cuts, replays, live streaming, and 3D ball or
camera reconstruction.

## What works now

- synchronized `annotated.mp4`, `tracks.jsonl`, `calibration.jsonl`, run metadata, and
  metrics;
- dynamic `H(t)` from semantic observations plus masked optical flow and RANSAC;
- `valid`, `predicted`, and `invalid` calibration states with explicit confidence and
  diagnostics;
- stable image-space IDs through short missed detections;
- raw and EMA-smoothed pitch tracks, reset on invalid calibration;
- adapters for local Ultralytics, exported SoccerNet DeepLab, and exported TVCalib
  predictors, with no runtime downloads;
- a licensed SoccerTrack v2 virtual-PTZ demo generator and versioned benchmark reports.

## Quickstart without private data or model weights

Python 3.11+ is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pitchtrack demo-fixture --out .tmp\fixture --camera-pan-px-per-frame 1.5
.\.venv\Scripts\python.exe -m pitchtrack analyze `
  --video .tmp\fixture\input.mp4 `
  --config .tmp\fixture\config.yaml `
  --calibration-mode hybrid_temporal `
  --out outputs\fixture-run
.\.venv\Scripts\python.exe -m pytest -q
```

PitchTrack writes pitch coordinates as `null` whenever calibration is invalid; it still
preserves image-space tracks.

## Real-model run

Install optional dependencies, acquire weights under their upstream terms, and keep the
files in the ignored local `weights/` directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[models]"
Copy-Item configs\models.example.yaml configs\models.local.yaml
.\.venv\Scripts\python.exe -m pitchtrack analyze `
  --video data\input\continuous-shot.mp4 `
  --config configs\models.local.yaml `
  --calibration-mode hybrid_temporal `
  --out outputs\match-run
```

The exported calibration predictors must return `image_points_px`, `pitch_points_m`, and
`confidences`. This interface lets a fine-tuned model replace a baseline without changing
the geometry pipeline.

## Public demo clip

After accepting/acquiring SoccerTrack v2 locally, keep it outside Git and run:

```powershell
.\.venv\Scripts\python.exe -m pitchtrack make-demo-clip `
  --source-root D:\datasets\soccertrack-v2 `
  --manifest configs\demo\soccertrack-v2-117093.yaml `
  --match-id 117093 --half 1 `
  --out outputs\demo
```

The source is CC BY 4.0 and the generated clip is explicitly labelled as a virtual PTZ
crop from fixed panoramic footage. See [licensing](docs/licensing.md).

## Architecture

```text
frame -> player detector -> image-space tracker --------------------+
  |                                                               |
  +-> semantic field observer -> RANSAC homography                 |
  +-> masked optical flow ------> temporal calibration manager     |
                                  | valid / predicted / invalid    |
                                  +-> pitch projection -> smoothing+
                                                           -> artifacts
```

Third-party components provide pretrained detections or field observations. PitchTrack's
original layer is the typed orchestration, moving-object masking, dynamic homography
propagation, temporal validation/reinitialisation, explicit invalid outputs, virtual-PTZ
provenance, artifact synchronization, and reproducible evaluation plumbing.

## Evaluation and limitations

See [evaluation](docs/evaluation.md) for metric definitions and exact commands, and
[portfolio presentation](docs/portfolio.md) for claims that are safe to make.

Current limitations:

- pretrained weights and gated benchmark videos are not bundled;
- the two baseline adapters require a local exported predictor matching the common
  landmark contract;
- the built-in tracker uses deterministic IoU association with lost-track retention and
  is not appearance/ReID tracking;
- official SoccerNet JaC@5, completeness, and HOTA remain unreported until the real
  validation assets and official evaluators are run;
- shot boundaries, replay detection, and live FPS/latency/dropped-frame behavior are
  planned after the offline benchmark is established.

No command pushes, uploads, or downloads data automatically.
