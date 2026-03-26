"""
Prepare YOLO dataset:
1. Delete augmented images and labels
2. Move unlabeled images to a separate folder
3. Split remaining train data into train/val sets
"""

import os
import random
import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).parent
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_VAL = DATASET_DIR / "labels" / "val"

VAL_RATIO = 0.2
RANDOM_SEED = 42


def main():
    # Step 0: Move val back to train
    IMAGES_VAL.mkdir(parents=True, exist_ok=True)
    LABELS_VAL.mkdir(parents=True, exist_ok=True)
    moved_back = 0
    for img in list(IMAGES_VAL.glob("*.*")):
        shutil.move(str(img), IMAGES_TRAIN / img.name)
        moved_back += 1
    for lbl in list(LABELS_VAL.glob("*.txt")):
        shutil.move(str(lbl), LABELS_TRAIN / lbl.name)
    print(f"Moved {moved_back} images back from val to train.")

    # Step 1: Delete augmented images and labels
    for d in [IMAGES_TRAIN, LABELS_TRAIN]:
        for f in d.glob("*_aug_*"):
            f.unlink()
    print(f"Deleted augmented files.")

    # Step 2: Create empty labels for unlabeled images (negative samples)
    label_stems = {p.stem for p in LABELS_TRAIN.glob("*.txt")}
    image_files = list(IMAGES_TRAIN.glob("*.*"))

    created = 0
    for img in image_files:
        if img.stem not in label_stems:
            (LABELS_TRAIN / (img.stem + ".txt")).write_text("")
            created += 1

    print(f"Created {created} empty labels for negative samples. Total images: {len(image_files)}")

    # Step 2: Split into train/val
    IMAGES_VAL.mkdir(parents=True, exist_ok=True)
    LABELS_VAL.mkdir(parents=True, exist_ok=True)

    labeled_images = sorted(IMAGES_TRAIN.glob("*.*"))
    random.seed(RANDOM_SEED)
    random.shuffle(labeled_images)

    val_count = max(1, int(len(labeled_images) * VAL_RATIO))
    val_images = labeled_images[:val_count]

    for img in val_images:
        label = LABELS_TRAIN / (img.stem + ".txt")
        shutil.move(str(img), IMAGES_VAL / img.name)
        if label.exists():
            shutil.move(str(label), LABELS_VAL / label.name)

    print(f"Moved {val_count} images to val. "
          f"Train: {len(labeled_images) - val_count}, Val: {val_count}")


if __name__ == "__main__":
    main()
