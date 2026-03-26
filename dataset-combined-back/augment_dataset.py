
"""
Augment YOLO keypoint dataset with 8 transformations per image:
1. Brightness + contrast shift
2. Gaussian noise
3. Horizontal flip (updates bbox + keypoint coords)
4. Slight rotation (±10°)
5. HSV / color jitter
6. Gaussian blur
7. Large hue shift (recolor dart shaft to random color)
8. Vertical flip (updates bbox + keypoint coords)
"""

import copy
import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DATASET_DIR = Path(__file__).parent
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --- Label helpers ---

def parse_label(label_path: Path) -> list[list[float]]:
    """Parse YOLO keypoint label: class x y w h [kp_x kp_y kp_v ...]"""
    rows = []
    for line in label_path.read_text().strip().splitlines():
        rows.append([float(v) for v in line.split()])
    return rows


def save_label(rows: list[list[float]], path: Path):
    lines = []
    for row in rows:
        parts = [str(int(row[0]))] + [f"{v:.6f}" for v in row[1:]]
        lines.append(" ".join(parts))
    path.write_text("\n".join(lines) + "\n")


def flip_labels_h(rows: list[list[float]]) -> list[list[float]]:
    """Horizontal flip: mirror x coords (normalized 0-1)."""
    out = copy.deepcopy(rows)
    for row in out:
        row[1] = 1.0 - row[1]  # bbox center x
        # keypoints start at index 5, stride 3 (x, y, v)
        for i in range(5, len(row), 3):
            if row[i + 2] > 0:  # only if visible
                row[i] = 1.0 - row[i]  # kp x
    return out


def flip_labels_v(rows: list[list[float]]) -> list[list[float]]:
    """Vertical flip: mirror y coords (normalized 0-1).
    This produces darts at unusual angles the model hasn't seen."""
    out = copy.deepcopy(rows)
    for row in out:
        row[2] = 1.0 - row[2]  # bbox center y
        for i in range(5, len(row), 3):
            if row[i + 2] > 0:
                row[i + 1] = 1.0 - row[i + 1]  # kp y
    return out


def rotate_labels(rows: list[list[float]], angle_deg: float,
                  img_w: int, img_h: int) -> list[list[float]]:
    """Rotate bbox centers and keypoints around image center in pixel space."""
    out = copy.deepcopy(rows)
    # Image center in pixels
    pcx, pcy = img_w / 2.0, img_h / 2.0
    rad = np.deg2rad(-angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    def rotate_point(nx, ny):
        """Rotate a normalized coord around image center via pixel space."""
        px, py = nx * img_w - pcx, ny * img_h - pcy
        rx = px * cos_a - py * sin_a + pcx
        ry = px * sin_a + py * cos_a + pcy
        return rx / img_w, ry / img_h

    for row in out:
        # Rotate bbox center
        row[1], row[2] = rotate_point(row[1], row[2])

        # Recompute axis-aligned bbox size from rotated corners
        bw_px, bh_px = row[3] * img_w, row[4] * img_h
        abs_cos, abs_sin = abs(cos_a), abs(sin_a)
        new_bw = bw_px * abs_cos + bh_px * abs_sin
        new_bh = bw_px * abs_sin + bh_px * abs_cos
        row[3] = new_bw / img_w
        row[4] = new_bh / img_h

        # Rotate keypoints
        for i in range(5, len(row), 3):
            if row[i + 2] > 0:
                row[i], row[i + 1] = rotate_point(row[i], row[i + 1])
    return out


# --- Image augmentations ---

def aug_brightness_contrast(img: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.7, 1.3)  # contrast
    beta = random.randint(-30, 30)     # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def aug_gaussian_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def aug_horizontal_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def aug_rotate(img: np.ndarray, angle: float, black_border: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = cv2.BORDER_CONSTANT if black_border else cv2.BORDER_REFLECT_101
    return cv2.warpAffine(img, M, (w, h), borderMode=border, borderValue=(0, 0, 0))


def aug_hsv_jitter(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30, 30), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-30, 30), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_dart_recolor(img: np.ndarray, labels: list[list[float]]) -> np.ndarray:
    """Shift hue only inside dart bounding boxes so the board stays natural.
    Each dart gets its own random hue shift."""
    h, w = img.shape[:2]
    result = img.copy()

    for row in labels:
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]

        # Bbox with 5% padding for feathered edge
        pad = 0.05
        x1 = max(0, int((cx - bw / 2 * (1 + pad)) * w))
        y1 = max(0, int((cy - bh / 2 * (1 + pad)) * h))
        x2 = min(w, int((cx + bw / 2 * (1 + pad)) * w))
        y2 = min(h, int((cy + bh / 2 * (1 + pad)) * h))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        # Shift hue in this region
        region = result[y1:y2, x1:x2]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.int16)
        shift = random.randint(30, 150)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Feathered mask: fully shifted in center, blended at edges
        rh, rw = y2 - y1, x2 - x1
        mask = np.ones((rh, rw), dtype=np.float32)
        border = max(3, min(rw, rh) // 6)
        for i in range(border):
            fade = (i + 1) / border
            mask[i, :] = min(mask[i, 0], fade)
            mask[rh - 1 - i, :] = min(mask[rh - 1 - i, 0], fade)
            mask[:, i] = np.minimum(mask[:, i], fade)
            mask[:, rw - 1 - i] = np.minimum(mask[:, rw - 1 - i], fade)

        mask_3ch = mask[:, :, np.newaxis]
        blended = (shifted * mask_3ch + region * (1 - mask_3ch)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended

    return result


def aug_vertical_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)


def aug_gaussian_blur(img: np.ndarray) -> np.ndarray:
    ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# --- Main ---

AUGMENTATIONS = [
    ("bright", aug_brightness_contrast, None),
    ("noise", aug_gaussian_noise, None),
    ("hflip", aug_horizontal_flip, "hflip"),
    ("rot", None, "rotate"),          # ±30°
    ("rot_wide", None, "rotate_wide"),  # ±45-90°
    ("hsv", aug_hsv_jitter, None),
    ("blur", aug_gaussian_blur, None),
    ("recolor", None, "recolor"),
    ("vflip", aug_vertical_flip, "vflip"),
]


def main():
    image_files = sorted(IMAGES_TRAIN.glob("*.png")) + sorted(IMAGES_TRAIN.glob("*.jpg"))
    total = 0

    for img_path in tqdm(image_files):
        label_path = LABELS_TRAIN / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        labels = parse_label(label_path)

        for aug_name, aug_fn, label_mode in AUGMENTATIONS:
            suffix = f"_aug_{aug_name}"
            out_img_path = IMAGES_TRAIN / f"{img_path.stem}{suffix}.jpg"
            out_lbl_path = LABELS_TRAIN / f"{img_path.stem}{suffix}.txt"

            if label_mode == "hflip":
                aug_img = aug_fn(img)
                aug_labels = flip_labels_h(labels)
            elif label_mode == "vflip":
                aug_img = aug_fn(img)
                aug_labels = flip_labels_v(labels)
            elif label_mode == "recolor":
                aug_img = aug_dart_recolor(img, labels)
                aug_labels = copy.deepcopy(labels)
            elif label_mode == "rotate":
                angle = random.uniform(-30, 30)
                aug_img = aug_rotate(img, angle)
                aug_labels = rotate_labels(labels, angle, w, h)
            elif label_mode == "rotate_wide":
                angle = random.choice([-1, 1]) * random.uniform(45, 90)
                aug_img = aug_rotate(img, angle, black_border=True)
                aug_labels = rotate_labels(labels, angle, w, h)
            else:
                aug_img = aug_fn(img)
                aug_labels = copy.deepcopy(labels)

            cv2.imwrite(str(out_img_path), aug_img)
            save_label(aug_labels, out_lbl_path)
            total += 1

    print(f"Generated {total} augmented images ({total // len(AUGMENTATIONS)} originals x {len(AUGMENTATIONS)} augmentations)")


if __name__ == "__main__":
    main()
