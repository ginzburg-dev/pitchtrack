# Licensing and demo assets

PitchTrack source code does not bundle football video, model weights, or private SoccerNet data.

SoccerTrack v2 GSR annotations are consumed with an incremental parser because the
released files use a large SoccerNet-COCO object. Derived media must say whether it is
model output or an annotation-backed reference render.

The planned public demo uses SoccerTrack v2 match `117093`, first half. The dataset publisher releases its videos and annotations under CC BY 4.0. Generated demo material must credit:

> SoccerTrack v2 — Atom Scott, Ikuma Uchida, Kento Kuroda, Yufi Kim, and Keisuke Fujii, arXiv:2508.01802. Dataset video and annotations licensed CC BY 4.0.

The public overlay and README must disclose that the presentation video is a **virtual PTZ crop derived from fixed panoramic footage**. It is not evidence of physical-camera motion. Physical moving-camera results are evaluated separately on SoccerNet assets under SoccerNet's access terms and are not redistributed.

Ultralytics is an optional integration and has AGPL-3.0/enterprise licensing. Review the selected distribution model before publishing binaries or offering a hosted service.

Model weights are never committed or downloaded at runtime. Every adapter requires an
explicit existing local path. Review the upstream SoccerNet, TVCalib, and detector
weight licenses before placing artifacts in the ignored `weights/` directory.
