<div align="center">
  <img src="./assets/logo.png" width="120" alt="OpenHoopVision logo" />
</div>
<h3 align="center"><b>OpenHoopVision</b></h3>
<p align="center">
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-blue?style=flat-square" alt="中文 README" /></a>
</p>

**TL;DR:** OpenHoopVision brings free AI tools and game data to amateur and youth basketball, making them accessible to practitioners, coaches, and their families. Its extensible framework supports new events, data, and applications. Exported tracking data can power further AI analysis to help teams understand games and improve performance.

## Demo

<p align="center">
  <img src="./assets/demo_17.gif" alt="Open Demo 17" width="410" />
  <img src="./assets/demo_26.gif" alt="Open Demo 26" width="410" />
</p>
<p align="center">
  <img src="./assets/demo_29.gif" alt="Open Demo 29" width="410" />
  <img src="./assets/demo_30.gif" alt="Open Demo 30" width="410" />
</p>

## What Is OpenHoopVision?

OpenHoopVision analyzes a basketball video one frame at a time. It detects players, the ball, and court features; follows players across frames; assigns team labels and global player IDs; and infers ball actions and possession events. The run produces an annotated video and a JSONL file with frame and event records.

## What Can It Do?

- Detect players, the ball, the rim, scorers, holders, and referees.
- Associate player detections with pose keypoints and local tracks.
- Estimate team colors and reconcile player identities across frames with sports ReID and jersey-number OCR.
- Map detected court keypoints to a court layout and draw a mini-map.
- Infer shots, dribbles, passes, rebounds, turnovers, and steals when the video provides enough evidence.
- Render player labels, event status, a scoreboard, and court overlays into an H.264 video.

## How Does It Work?

The video pipeline observes each frame, combines detector and pose results with local tracking, then updates player identity and event state. The renderer and JSONL writer use the same frame context so the visual output and recorded data stay aligned.

| Stage | Responsibility |
| --- | --- |
| Detection and pose | Find basketball entities and estimate player and court keypoints. |
| Local tracking | Keep player and ball tracks stable between frames. |
| Identity and teams | Match player crops across time, classify teams, and collect jersey-number reads. |
| Event inference | Track ball actions and possession changes, resolve overlaps, attribute shots, and update the scoreboard. |
| Output | Draw overlays and write an annotated video plus per-frame JSONL records. |

### Shot Event

OpenHoopVision uses specialized detectors for ball actions and possession changes, then resolves overlaps in a shared timeline. Resolved shot events take priority over overlapping passes. Passes require visible ball-transfer evidence and stable receiver control; when the sender's team is known, it must match the receiver's team.

The shot detector uses ball motion and rim evidence to distinguish shots from dribbles. It reports outcomes as `made`, `miss`, or `unknown`. Made shots use court location for a two- or three-point value when reliable; otherwise, the system applies a labeled two-point fallback.

## Performance and Model Metrics

| Metric | Result |
| --- | --- |
| RF-DETR detector (`Basketball-Detection-v6`) | mAP@0.5: 84.9%; mAP@0.5:0.95: 62.3% |
| YOLO26s court pose (`court-detection-v2`) | mAP@0.5: 96.8%; mAP@0.5:0.95: 81.1% |
| End-to-end shot pipeline (`basketball-event-recognition`) | Precision: 85.51%; recall: 83.10%; F1: 84.29%; team attribution accuracy: 79.66% (47/59). |

All datasets were curated and annotated in-house. Model scores use validation splits. End-to-end metrics cover 71 shots across 7 videos, matched one-to-one at temporal IoU ≥ 0.5. A ±0.5 s onset tolerance was used for one incomplete annotation.

## Run OpenHoopVision

### Requirements

- Python 3.10 or newer.
- FFmpeg is required for H.264 video output. Install a platform build from the [official download page](https://ffmpeg.org/download.html) and make sure `ffmpeg` is available on `PATH`.
- The model checkpoint files listed below. The program reads the files at the given paths and does not download them automatically.
- An NVIDIA GPU is recommended for practical processing speed. When CUDA is unavailable, the runner selects CPU inference.

Install the project and its dependencies from the repository root:

```bash
python -m pip install -e .
```

`requirements.lock` contains the direct package pins for the validated CUDA 13.0 environment. To use those versions instead:

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

### Model Checkpoints

The default checkpoint paths are relative to the project directory:

| Component | Default path |
| --- | --- |
| Basketball detector | `weights/rf-detr-basketball.pth` |
| Player pose | `weights/yolov8n-pose.pt` |
| Court pose | `weights/yolo26s-pose-court-33kp.pt` |
| Player ReID | [`weights/sports_model.pth.tar-60`](https://github.com/sjc042/gta-link/tree/main/reid_checkpoints) |
| Jersey-number OCR | `weights/basketball_jersey_number.ckpt` |

### Analyze a Video

Run from the project directory and provide an input video with `--video`:

```bash
python main.py --video {your_video}.mp4
```

`python main.py --help` lists all detection thresholds, tracking settings, model paths, and output options.

### Output Files

- `--output` writes an annotated H.264 MP4. The default is `../output.mp4` when run from the project directory.
- `--tracks` writes one JSON object per frame. The default is `../tracks.jsonl`.

Each JSONL frame record uses the `basketball_frame_v2` schema and includes the frame number, timestamp, players, holder, ball and rim features, raw detections, and events. The first record also contains the run configuration. Event records use frame intervals and include event-specific fields such as shot result and points, pass receiver, or rebound type.

## Project Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Reads command-line options and starts a video analysis run. |
| `pipeline/` | Coordinates frame-by-frame processing and passes results between stages. |
| `models/` | Loads the vision models and adapts their outputs for the analysis pipeline. |
| `tracking/` | Follows players and the ball across frames, and estimates teams and court positions. |
| `reid/` | Matches player identities across tracks and uses jersey-number reads as supporting evidence. |
| `events/` | Detects basketball actions and possession changes, then attributes and scores events. |
| `render/` | Draws analysis results on frames and writes the annotated video. |
| `assets/` | Stores the project logo and the demo GIF previews shown in this README. |

## Current Scope

The system currently infers shots, dribbles, passes, rebounds, turnovers, and steals. Assists, blocks, and fouls are not supported; unreliable court locations trigger a labeled two-point fallback.

Known limitations: Occlusion, camera cuts, and low-resolution footage can make ball tracking, player identity, and event results unreliable. Low camera angles that obscure court boundaries can prevent the mini-map from working. The pipeline runs offline on recorded video, can be slow without a capable GPU, and requires model checkpoints to be installed separately.

Next, I'm exploring free tools built on this project, such as automatic video highlights. 

Coming soon...

## References

[RF-DETR](https://github.com/roboflow/rf-detr) — object-detection framework used to load the basketball detector checkpoint.

[Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO runtime for the [YOLOv8 pose](https://docs.ultralytics.com/models/yolov8/) player checkpoint and [YOLO26](https://docs.ultralytics.com/models/yolo26/) court-pose checkpoint.

[McByte++](https://github.com/tstanczyk95/McBytePlusPlus) — acknowledged as a source for the local tracking implementation and the sports ReID checkpoint.

[ByteTrack](https://github.com/FoundationVision/ByteTrack) — high/low-confidence association and Kalman-filter foundations used by the local tracker.

[OSNet / Torchreid](https://github.com/KaiyangZhou/deep-person-reid) — the OSNet x1.0 architecture adapted in [`models/appearance.py`](models/appearance.py).

[PARSeq](https://github.com/baudm/parseq) — OCR architecture adapted in [`models/jersey_number.py`](models/jersey_number.py) for jersey-number recognition.

## License

OpenHoopVision's original code and project-trained model artifacts are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Third-party models, code, and dependencies remain subject to their respective licenses, including [Ultralytics' AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). The demo GIFs are previews derived from videos found online; the original footage remains subject to its respective copyright and terms.
