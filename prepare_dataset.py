from __future__ import annotations

import random
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "dataset" / "dataset"
OUT_DIR = ROOT / "dataset_split"
SEED = 42
VAL_RATIO = 0.2


def source_group(image_path: Path) -> str:
    """Group likely-related frames/images to reduce train/val leakage."""
    stem = image_path.stem
    stem = stem.split("_jpg.rf.")[0]
    stem = stem.split(".rf.")[0]
    stem = re.sub(r"-?train_car_\d+$", "-train_car", stem)
    stem = re.sub(r"mpv\d+$", "mpv", stem)
    return stem


def has_object(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def copy_pair(image_path: Path, split: str) -> None:
    label_path = RAW_DIR / "labels" / f"{image_path.stem}.txt"
    image_dst = OUT_DIR / "images" / split / image_path.name
    label_dst = OUT_DIR / "labels" / split / label_path.name
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_dst)
    if label_path.exists():
        shutil.copy2(label_path, label_dst)
    else:
        label_dst.write_text("", encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    image_paths = sorted((RAW_DIR / "images").glob("*.*"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {RAW_DIR / 'images'}")

    positives_by_group: dict[str, list[Path]] = {}
    negatives: list[Path] = []
    for image_path in image_paths:
        label_path = RAW_DIR / "labels" / f"{image_path.stem}.txt"
        if has_object(label_path):
            positives_by_group.setdefault(source_group(image_path), []).append(image_path)
        else:
            negatives.append(image_path)

    groups = list(positives_by_group.values())
    random.shuffle(groups)
    random.shuffle(negatives)

    val_target_pos = round(sum(len(group) for group in groups) * VAL_RATIO)
    val_images: list[Path] = []
    train_images: list[Path] = []
    val_pos = 0
    for group in groups:
        if val_pos < val_target_pos:
            val_images.extend(group)
            val_pos += len(group)
        else:
            train_images.extend(group)

    val_target_neg = round(len(negatives) * VAL_RATIO)
    val_images.extend(negatives[:val_target_neg])
    train_images.extend(negatives[val_target_neg:])

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for image_path in train_images:
        copy_pair(image_path, "train")
    for image_path in val_images:
        copy_pair(image_path, "val")

    data_yaml = OUT_DIR / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {OUT_DIR.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "nc: 1",
                "names: ['car_front']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Created {OUT_DIR}")
    print(f"train images: {len(train_images)}")
    print(f"val images: {len(val_images)}")
    print(f"train negatives: {sum(not has_object(RAW_DIR / 'labels' / (p.stem + '.txt')) for p in train_images)}")
    print(f"val negatives: {sum(not has_object(RAW_DIR / 'labels' / (p.stem + '.txt')) for p in val_images)}")


if __name__ == "__main__":
    main()
