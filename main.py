import argparse
import os
from pathlib import Path
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, default=WEIGHTS_DIR / "rf-detr-basketball.pth")
    parser.add_argument("--pose-model", type=Path, default=WEIGHTS_DIR / "yolov8n-pose.pt")
    parser.add_argument("--reid-model", type=Path, default=WEIGHTS_DIR / "sports_model.pth.tar-60")
    parser.add_argument("--jersey-ocr-model", type=Path, default=WEIGHTS_DIR / "basketball_jersey_number.ckpt")
    parser.add_argument("--court-model", type=Path, default=WEIGHTS_DIR / "yolo26s-pose-court-33kp.pt")
    parser.add_argument("--output", type=Path, default=WORKSPACE_ROOT / 'output.mp4')
    parser.add_argument("--tracks", type=Path, default=WORKSPACE_ROOT / 'tracks.jsonl')
    parser.add_argument("--player-threshold", type=float, default=0.41)
    parser.add_argument("--ball-threshold", type=float, default=0.34)
    parser.add_argument("--rim-threshold", type=float, default=0.5)
    parser.add_argument("--holder-threshold", type=float, default=0.38)
    parser.add_argument("--scorer-threshold", type=float, default=0.38)
    parser.add_argument("--referee-threshold", type=float, default=0.54)
    parser.add_argument("--short-side", type=int, default=640)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--trail-length", type=int, default=16)
    parser.add_argument("--pose-imgsz", type=int, default=1280)
    parser.add_argument("--pose-interval", type=int, default=1)
    parser.add_argument("--team-cache-ttl-seconds", type=float, default=10.0)
    parser.add_argument("--court-imgsz", type=int, default=640)
    parser.add_argument("--court-interval", type=int, default=5)
    parser.add_argument("--holder-hold-frames", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["LOG_LEVEL"] = "ERROR"
    from pipeline.video_analysis import VideoAnalysisRunner

    VideoAnalysisRunner(_parse_args() if args is None else args).run()


if __name__ == "__main__":
    main()
