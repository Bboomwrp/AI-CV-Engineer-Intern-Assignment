from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run car_front inference on Video.mp4.")
    parser.add_argument("--weights", default="best.pt", help="Fine-tuned weights.")
    parser.add_argument("--source", default="Video.mp4", help="Input video.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0, help="Use 0 for CUDA GPU, cpu for CPU.")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="video_inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = ROOT / args.weights
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}. Run train.py first or pass --weights.")

    model = YOLO(str(weights))
    results = model.predict(
        source=str(ROOT / args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(ROOT / args.project),
        name=args.name,
        exist_ok=True,
    )

    save_dir = Path(results[0].save_dir) if results else ROOT / args.project / args.name
    produced = list(save_dir.glob("*.mp4")) + list(save_dir.glob("*.avi"))
    if produced:
        output = ROOT / "output_video.mp4"
        produced[0].replace(output)
        print(f"Output video: {output}")
    else:
        print(f"Inference outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
