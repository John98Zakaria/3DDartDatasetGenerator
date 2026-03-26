"""
Copy-Paste Synthetic Data Generator for YOLO Keypoint Dart Detection.

Crops darts from labeled images using bounding boxes, creates masks by
comparing against the empty board, and pastes them onto the empty board
at random positions to generate synthetic training images with labels.

Usage:
    python copy_paste_synthetic.py [--num-images 500] [--max-darts 3] [--output-dir synthetic]
"""

import argparse
import copy
import random
from pathlib import Path

import cv2
import numpy as np

DATASET_DIR = Path(__file__).parent
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"

# All 3 camera views of the empty board
EMPTY_BOARD_STEMS = [
    "2026-03-08T20_18_50.700Z_0x202331000bda5844_current",
    "2026-03-08T20_18_50.700Z_0x202332000bda5844_current",
    "2026-03-08T20_18_50.700Z_0x202333000bda5844_current",
]

RANDOM_SEED = 123
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --- Label helpers ---

def parse_label(label_path: Path) -> list[list[float]]:
    rows = []
    text = label_path.read_text().strip()
    if not text:
        return rows
    for line in text.splitlines():
        rows.append([float(v) for v in line.split()])
    return rows


def save_label(rows: list[list[float]], path: Path):
    lines = []
    for row in rows:
        parts = [str(int(row[0]))] + [f"{v:.6f}" for v in row[1:]]
        lines.append(" ".join(parts))
    path.write_text("\n".join(lines) + "\n")


# --- Dart extraction ---

def extract_dart_crops(
    empty_boards: dict[str, np.ndarray],
) -> list[dict]:
    """Extract dart crops from all labeled training images.

    Returns list of dicts with keys:
        crop: BGRA image (with alpha mask)
        tip_offset: (dx, dy) of tip keypoint relative to crop top-left, normalized to crop size
        base_offset: (dx, dy) of base keypoint relative to crop top-left, normalized to crop size
        tip_vis, base_vis: visibility flags
        aspect_ratio: w/h of the original bbox
    """
    crops = []
    image_files = sorted(IMAGES_TRAIN.glob("*.jpg"))

    for img_path in image_files:
        # Skip augmented images and empty boards
        if "_aug_" in img_path.stem:
            continue
        if img_path.stem in EMPTY_BOARD_STEMS:
            continue

        label_path = LABELS_TRAIN / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        labels = parse_label(label_path)
        if not labels:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Find the matching empty board by camera ID
        camera_id = _extract_camera_id(img_path.stem)
        empty = empty_boards.get(camera_id)
        if empty is None:
            continue

        for row in labels:
            # Parse YOLO format: class cx cy bw bh kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
            cx, cy, bw, bh = row[1], row[2], row[3], row[4]
            tip_x, tip_y, tip_v = row[5], row[6], row[7]
            base_x, base_y, base_v = row[8], row[9], row[10]

            # Skip darts with very small bboxes or out-of-frame keypoints
            if bw < 0.01 or bh < 0.01:
                continue
            if tip_v < 2 and base_v < 2:
                continue

            # Convert to pixel coords with padding
            pad_factor = 0.15  # 15% padding around bbox
            px1 = int((cx - bw / 2 * (1 + pad_factor)) * w)
            py1 = int((cy - bh / 2 * (1 + pad_factor)) * h)
            px2 = int((cx + bw / 2 * (1 + pad_factor)) * w)
            py2 = int((cy + bh / 2 * (1 + pad_factor)) * h)

            # Clamp to image bounds
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            crop_w = px2 - px1
            crop_h = py2 - py1
            if crop_w < 10 or crop_h < 10:
                continue

            # Crop from source and empty board
            dart_region = img[py1:py2, px1:px2].copy()
            empty_region = empty[py1:py2, px1:px2].copy()

            # Create mask via absolute difference
            diff = cv2.absdiff(dart_region, empty_region)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Threshold + morphological cleanup
            _, mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Dilate slightly to include edges
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Feather the mask edges for smoother blending
            mask = cv2.GaussianBlur(mask, (7, 7), 0)

            # If mask covers less than 5% of the crop, skip (probably noise)
            if mask.sum() / 255 < 0.05 * crop_w * crop_h:
                continue

            # Create BGRA crop
            bgra = cv2.cvtColor(dart_region, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = mask

            # Compute keypoint offsets relative to crop (normalized 0-1 within crop)
            tip_px = tip_x * w - px1
            tip_py = tip_y * h - py1
            base_px = base_x * w - px1
            base_py = base_y * h - py1

            crops.append({
                "crop": bgra,
                "tip_offset": (tip_px / crop_w, tip_py / crop_h),
                "base_offset": (base_px / crop_w, base_py / crop_h),
                "tip_vis": int(tip_v),
                "base_vis": int(base_v),
                "aspect_ratio": crop_w / crop_h,
            })

    return crops


def _extract_camera_id(stem: str) -> str:
    """Extract camera hex ID from filename like '..._0x202333000bda5844_current'."""
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p.startswith("0x"):
            return p
    return ""


# --- Synthetic image generation ---

def paste_dart(
    canvas: np.ndarray,
    dart: dict,
    center_x: int,
    center_y: int,
    scale: float = 1.0,
    angle: float = 0.0,
) -> dict | None:
    """Paste a dart crop onto canvas at given position with scale and rotation.

    Returns YOLO label dict or None if paste failed.
    """
    h_canvas, w_canvas = canvas.shape[:2]
    crop = dart["crop"].copy()
    crop_h, crop_w = crop.shape[:2]

    # Random hue shift on just the dart (not the background) — 50% chance
    if random.random() < 0.5:
        bgr = crop[:, :, :3]
        alpha = crop[:, :, 3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(0, 179)) % 180
        crop[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Scale the crop
    new_w = max(1, int(crop_w * scale))
    new_h = max(1, int(crop_h * scale))
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Rotate if needed
    if abs(angle) > 0.5:
        rot_mat = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
        # Compute new bounding size after rotation
        cos_a = abs(rot_mat[0, 0])
        sin_a = abs(rot_mat[0, 1])
        rot_w = int(new_h * sin_a + new_w * cos_a)
        rot_h = int(new_h * cos_a + new_w * sin_a)
        rot_mat[0, 2] += (rot_w - new_w) / 2
        rot_mat[1, 2] += (rot_h - new_h) / 2
        crop = cv2.warpAffine(crop, rot_mat, (rot_w, rot_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        new_w, new_h = rot_w, rot_h

    # Compute paste position (center_x, center_y is where dart center goes)
    x1 = center_x - new_w // 2
    y1 = center_y - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Compute visible region (clip to canvas)
    vis_x1 = max(0, x1)
    vis_y1 = max(0, y1)
    vis_x2 = min(w_canvas, x2)
    vis_y2 = min(h_canvas, y2)

    if vis_x2 - vis_x1 < 5 or vis_y2 - vis_y1 < 5:
        return None

    # Crop region that maps to visible canvas area
    crop_vis_x1 = vis_x1 - x1
    crop_vis_y1 = vis_y1 - y1
    crop_vis_x2 = crop_vis_x1 + (vis_x2 - vis_x1)
    crop_vis_y2 = crop_vis_y1 + (vis_y2 - vis_y1)

    crop_region = crop[crop_vis_y1:crop_vis_y2, crop_vis_x1:crop_vis_x2]
    alpha = crop_region[:, :, 3:4].astype(np.float32) / 255.0
    bgr = crop_region[:, :, :3].astype(np.float32)

    canvas_region = canvas[vis_y1:vis_y2, vis_x1:vis_x2].astype(np.float32)
    blended = bgr * alpha + canvas_region * (1 - alpha)
    canvas[vis_y1:vis_y2, vis_x1:vis_x2] = blended.astype(np.uint8)

    # Compute keypoints in canvas coordinates
    # First, map original crop-relative keypoints through scale and rotation
    tip_cx = dart["tip_offset"][0] * crop_w * scale
    tip_cy = dart["tip_offset"][1] * crop_h * scale
    base_cx = dart["base_offset"][0] * crop_w * scale
    base_cy = dart["base_offset"][1] * crop_h * scale

    if abs(angle) > 0.5:
        # Apply rotation to keypoints
        for kp in [(tip_cx, tip_cy), (base_cx, base_cy)]:
            pass  # will transform below
        origin_x = crop_w * scale / 2
        origin_y = crop_h * scale / 2
        rad = np.deg2rad(-angle)
        cos_r, sin_r = np.cos(rad), np.sin(rad)

        def rotate_point(px, py):
            dx, dy = px - origin_x, py - origin_y
            rx = dx * cos_r - dy * sin_r + new_w / 2
            ry = dx * sin_r + dy * cos_r + new_h / 2
            return rx, ry

        tip_cx, tip_cy = rotate_point(tip_cx, tip_cy)
        base_cx, base_cy = rotate_point(base_cx, base_cy)

    # Map to canvas coordinates
    tip_canvas_x = x1 + tip_cx
    tip_canvas_y = y1 + tip_cy
    base_canvas_x = x1 + base_cx
    base_canvas_y = y1 + base_cy

    # Compute tight bounding box from alpha mask on canvas
    alpha_2d = crop_region[:, :, 3]
    ys, xs = np.where(alpha_2d > 30)
    if len(xs) < 5:
        return None

    bbox_x1 = vis_x1 + xs.min()
    bbox_y1 = vis_y1 + ys.min()
    bbox_x2 = vis_x1 + xs.max()
    bbox_y2 = vis_y1 + ys.max()

    # Convert to YOLO normalized format
    bbox_cx = (bbox_x1 + bbox_x2) / 2 / w_canvas
    bbox_cy = (bbox_y1 + bbox_y2) / 2 / h_canvas
    bbox_w = (bbox_x2 - bbox_x1) / w_canvas
    bbox_h = (bbox_y2 - bbox_y1) / h_canvas

    # Determine keypoint visibility (2 if on canvas, 1 if off-canvas)
    tip_vis = 2 if (0 <= tip_canvas_x < w_canvas and 0 <= tip_canvas_y < h_canvas) else 1
    base_vis = 2 if (0 <= base_canvas_x < w_canvas and 0 <= base_canvas_y < h_canvas) else 1

    # Use original visibility if it was already occluded
    if dart["tip_vis"] < 2:
        tip_vis = dart["tip_vis"]
    if dart["base_vis"] < 2:
        base_vis = dart["base_vis"]

    # Clamp keypoints to [0, 1]
    tip_nx = np.clip(tip_canvas_x / w_canvas, 0, 1)
    tip_ny = np.clip(tip_canvas_y / h_canvas, 0, 1)
    base_nx = np.clip(base_canvas_x / w_canvas, 0, 1)
    base_ny = np.clip(base_canvas_y / h_canvas, 0, 1)

    return [
        0,  # class
        bbox_cx, bbox_cy, bbox_w, bbox_h,
        tip_nx, tip_ny, float(tip_vis),
        base_nx, base_ny, float(base_vis),
    ]


def generate_synthetic_image(
    empty_boards: list[np.ndarray],
    dart_crops: list[dict],
    num_darts: int,
) -> tuple[np.ndarray, list[list[float]]]:
    """Generate one synthetic image by pasting darts onto an empty board."""
    # Pick a random empty board
    bg = random.choice(empty_boards).copy()
    h, w = bg.shape[:2]

    labels = []
    occupied = []  # track paste centers to avoid heavy overlap

    for _ in range(num_darts):
        dart = random.choice(dart_crops)

        # Random scale (0.8x - 1.2x)
        scale = random.uniform(0.8, 1.2)

        # Random rotation (±15°)
        angle = random.uniform(-15, 15)

        # Random position — bias toward the board area (roughly center 70% of image)
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        for attempt in range(20):
            cx = random.randint(margin_x, w - margin_x)
            cy = random.randint(margin_y, h - margin_y)
            # Check minimum distance from other darts
            too_close = False
            for ox, oy in occupied:
                if abs(cx - ox) < 60 and abs(cy - oy) < 60:
                    too_close = True
                    break
            if not too_close:
                break

        label = paste_dart(bg, dart, cx, cy, scale, angle)
        if label is not None:
            labels.append(label)
            occupied.append((cx, cy))

    # Apply random augmentations to the whole image
    if random.random() < 0.5:
        # Brightness/contrast
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        bg = cv2.convertScaleAbs(bg, alpha=alpha, beta=beta)

    if random.random() < 0.3:
        # Slight blur
        ksize = random.choice([3, 5])
        bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)

    if random.random() < 0.3:
        # HSV jitter
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-8, 8)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-20, 20), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-20, 20), 0, 255)
        bg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.2:
        # Gaussian noise
        noise = np.random.normal(0, 10, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg, labels


def main():
    parser = argparse.ArgumentParser(description="Copy-paste synthetic data generator")
    parser.add_argument("--num-images", type=int, default=500,
                        help="Number of synthetic images to generate")
    parser.add_argument("--max-darts", type=int, default=3,
                        help="Maximum darts per image")
    parser.add_argument("--output-dir", type=str, default="synthetic",
                        help="Output subdirectory name (under images/ and labels/)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    out_images = DATASET_DIR / "images" / args.output_dir
    out_labels = DATASET_DIR / "labels" / args.output_dir
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Load empty boards
    print("Loading empty boards...")
    empty_boards_by_camera = {}
    empty_boards_list = []
    for stem in EMPTY_BOARD_STEMS:
        img = cv2.imread(str(IMAGES_TRAIN / f"{stem}.jpg"))
        if img is not None:
            camera_id = _extract_camera_id(stem)
            empty_boards_by_camera[camera_id] = img
            empty_boards_list.append(img)
            print(f"  Loaded {stem} (camera {camera_id})")

    if not empty_boards_list:
        print("ERROR: No empty board images found!")
        return

    # Extract dart crops
    print("Extracting dart crops from labeled images...")
    dart_crops = extract_dart_crops(empty_boards_by_camera)
    print(f"  Extracted {len(dart_crops)} dart crops")

    if not dart_crops:
        print("ERROR: No dart crops extracted!")
        return

    # Generate synthetic images
    print(f"Generating {args.num_images} synthetic images...")
    dart_distribution = {
        0: 0.10,  # 10% empty (negative samples)
        1: 0.30,  # 30% single dart
        2: 0.35,  # 35% two darts
        3: 0.25,  # 25% three darts
    }

    for i in range(args.num_images):
        # Sample number of darts
        r = random.random()
        cumulative = 0
        num_darts = 1
        for n, prob in dart_distribution.items():
            cumulative += prob
            if r < cumulative:
                num_darts = min(n, args.max_darts)
                break

        img, labels = generate_synthetic_image(empty_boards_list, dart_crops, num_darts)

        name = f"synthetic_{i:05d}"
        cv2.imwrite(str(out_images / f"{name}.jpg"), img)
        save_label(labels, out_labels / f"{name}.txt")

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.num_images}")

    # Print stats
    total_labels = 0
    empty_count = 0
    for lbl_file in sorted(out_labels.glob("*.txt")):
        content = lbl_file.read_text().strip()
        if not content:
            empty_count += 1
        else:
            total_labels += len(content.splitlines())

    print(f"\nDone! Generated {args.num_images} images:")
    print(f"  Images: {out_images}")
    print(f"  Labels: {out_labels}")
    print(f"  Total dart annotations: {total_labels}")
    print(f"  Empty images (negative samples): {empty_count}")
    print(f"\nTo include in training, update dataset.yaml or merge into images/train + labels/train")


if __name__ == "__main__":
    main()
