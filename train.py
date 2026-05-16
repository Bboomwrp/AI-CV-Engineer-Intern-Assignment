from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import prepare_dataset


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26s for car_front detection.")
    parser.add_argument("--weights", default="yolo26s.pt", help="Base YOLO weights.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=0, help="Use 0 for CUDA GPU, cpu for CPU.")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="car_front_yolo26s")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset.main()

    model = YOLO(str(ROOT / args.weights))
    run = model.train(
        data=str(ROOT / "dataset_split" / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / args.project),
        name=args.name,
        exist_ok=True,
        patience=20,
        seed=42,
        deterministic=True,
        single_cls=True,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=3.0,
        translate=0.08,
        scale=0.4,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.6,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.2,
        auto_augment="randaugment",
    )

    run_dir = Path(run.save_dir)
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, ROOT / "best.pt")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    for pattern in ("results.*", "confusion_matrix*", "PR_curve*", "F1_curve*", "P_curve*", "R_curve*", "val_batch*_pred.jpg", "val_batch*_labels.jpg", "args.yaml"):
        for path in run_dir.glob(pattern):
            shutil.copy2(path, results_dir / path.name)

    print(f"Training run: {run_dir}")
    print(f"Best weights: {best}")
    print(f"Copied summary artifacts to: {results_dir}")


if __name__ == "__main__":
    main()
