"""¡¡
Copy-Paste Augmentation for Dart Dataset
=========================================
Uses SAM3 to segment darts from 1-dart images, then pastes them
onto empty board images to generate new training samples.
"""

import os
import glob
import random
import numpy as np
import cv2
from ultralytics import SAM

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset-combined"
SRC_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
SRC_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

OUT_DIR = "dataset-copypaste"
OUT_IMAGES_DIR = os.path.join(OUT_DIR, "images", "train")
OUT_LABELS_DIR = os.path.join(OUT_DIR, "labels", "train")
SAM3_PT = "sam3.pt"

# How many darts to place per generated image
MIN_DARTS = 1
MAX_DARTS = 6

# How many images to generate per empty board
IMAGES_PER_BOARD = 100

# Placement jitter: random scale range for pasted darts
SCALE_MIN = 0.8
SCALE_MAX = 1.2

SEED = 42


def find_images_by_dart_count(images_dir, labels_dir, target_count):
    """Find non-augmented images with exactly `target_count` darts.
    Empty boards (target_count=0) may have no label file at all.
    """
    results = []
    for img_path in glob.glob(os.path.join(images_dir, "*.jpg")):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # Skip augmented images
        if "_aug_" in basename:
            continue
        label_path = os.path.join(labels_dir, basename + ".txt")
        if not os.path.exists(label_path):
            # No label file means 0 darts
            if target_count == 0:
                results.append((img_path, label_path, []))
            continue
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) == target_count:
            results.append((img_path, label_path, lines))
    return results


def parse_yolo_label(line, img_w, img_h):
    """Parse a YOLO pose label line into pixel coordinates.
    Returns: class_id, bbox (x1,y1,x2,y2), keypoints [(x,y,vis), ...]
    """
    parts = list(map(float, line.split()))
    class_id = int(parts[0])
    cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
    # Convert normalized to pixel
    cx_px = cx * img_w
    cy_px = cy * img_h
    w_px = w * img_w
    h_px = h * img_h
    x1 = cx_px - w_px / 2
    y1 = cy_px - h_px / 2
    x2 = cx_px + w_px / 2
    y2 = cy_px + h_px / 2
    # Keypoints
    kps = []
    i = 5
    while i + 2 < len(parts):
        kp_x = parts[i] * img_w
        kp_y = parts[i + 1] * img_h
        kp_vis = parts[i + 2]
        kps.append((kp_x, kp_y, kp_vis))
        i += 3
    return class_id, (x1, y1, x2, y2), kps


def run_sam(model, img, bbox, points=None, point_labels=None):
    """Run SAM with box prompt and optional point prompts. Returns binary mask or None."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    kwargs = {"bboxes": [[x1, y1, x2, y2]]}
    if points and point_labels:
        kwargs["points"] = [points]
        kwargs["labels"] = [point_labels]
    results = model(img, **kwargs)
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        combined = np.zeros(masks.shape[1:], dtype=bool)
        for m in masks:
            combined |= m.astype(bool)
        return combined.astype(np.uint8) * 255
    return None


def make_overlay(img, mask, bbox, points=None, point_labels=None):
    """Create a visualization overlay: green mask, red bbox, colored points."""
    vis = img.copy()
    if mask is not None:
        green_overlay = np.zeros_like(vis)
        green_overlay[:, :, 1] = 255
        mask_bool = mask > 127
        vis[mask_bool] = cv2.addWeighted(vis, 0.5, green_overlay, 0.5, 0)[mask_bool]

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if points:
        for (px, py), lbl in zip(points, point_labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(vis, (px, py), 6, color, -1)
            cv2.circle(vis, (px, py), 6, (255, 255, 255), 1)

    # Instructions
    cv2.putText(vis, "LMB: include  RMB: exclude  R: reset  A: accept  S: skip  Q: quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis, "LMB: include  RMB: exclude  R: reset  A: accept  S: skip  Q: quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis


def interactive_segment(model, img, bbox, basename):
    """Interactively refine SAM segmentation with point prompts.
    Returns binary mask or None if skipped/quit.
    Returns 'quit' string if user wants to stop entirely.
    """
    points = []
    point_labels = []
    mask = run_sam(model, img, bbox)

    win_name = f"SAM - {basename}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, 700)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            point_labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y])
            point_labels.append(0)

    cv2.setMouseCallback(win_name, on_mouse)

    prev_len = 0
    while True:
        # Re-run SAM if new points were added
        if len(points) > prev_len:
            prev_len = len(points)
            mask = run_sam(model, img, bbox, points, point_labels)

        vis = make_overlay(img, mask, bbox, points, point_labels)
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(50) & 0xFF

        if key == ord('a'):  # accept
            cv2.destroyWindow(win_name)
            return mask
        elif key == ord('s'):  # skip
            cv2.destroyWindow(win_name)
            return None
        elif key == ord('q'):  # quit entirely
            cv2.destroyWindow(win_name)
            return 'quit'
        elif key == ord('r'):  # reset points
            points.clear()
            point_labels.clear()
            prev_len = 0
            mask = run_sam(model, img, bbox)


def segment_dart_with_sam(model, img, bbox):
    """Use SAM3 box prompt to get the dart mask. Returns binary mask."""
    return run_sam(model, img, bbox)


def extract_dart(img, mask, bbox):
    """Extract the dart region (RGBA) using the mask and bbox.
    Returns: cropped RGBA image, offset (x, y) of crop in original image.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = img.shape[:2]
    # Pad bbox slightly to include full mask
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop_img = img[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    # Create RGBA
    rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = crop_mask
    return rgba, (x1, y1)


def paste_dart(board_img, dart_rgba, target_cx, target_cy, scale=1.0):
    """Paste a dart onto the board at (target_cx, target_cy) with given scale.
    Returns: modified board, actual bbox (x1,y1,x2,y2) in pixels.
    """
    h_board, w_board = board_img.shape[:2]
    dh, dw = dart_rgba.shape[:2]

    # Scale
    new_w = int(dw * scale)
    new_h = int(dh * scale)
    if new_w < 5 or new_h < 5:
        return board_img, None
    dart_scaled = cv2.resize(dart_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute paste position (top-left)
    px = int(target_cx - new_w / 2)
    py = int(target_cy - new_h / 2)

    # Clamp to board boundaries
    src_x1 = max(0, -px)
    src_y1 = max(0, -py)
    src_x2 = min(new_w, w_board - px)
    src_y2 = min(new_h, h_board - py)

    dst_x1 = max(0, px)
    dst_y1 = max(0, py)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return board_img, None

    # Alpha blending
    alpha = dart_scaled[src_y1:src_y2, src_x1:src_x2, 3:4].astype(np.float32) / 255.0
    rgb = dart_scaled[src_y1:src_y2, src_x1:src_x2, :3].astype(np.float32)
    bg = board_img[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    board_img[dst_y1:dst_y2, dst_x1:dst_x2] = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

    return board_img, (dst_x1, dst_y1, dst_x2, dst_y2)


def transform_keypoints(kps, crop_offset, scale, target_cx, target_cy, dart_rgba_shape):
    """Transform keypoints from original image coords to pasted coords."""
    ox, oy = crop_offset
    dh, dw = dart_rgba_shape[:2]
    new_kps = []
    for kp_x, kp_y, kp_vis in kps:
        # Shift to crop-local coords
        lx = kp_x - ox
        ly = kp_y - oy
        # Scale
        lx *= scale
        ly *= scale
        # Shift to paste position
        new_w = dw * scale
        new_h = dh * scale
        px = target_cx - new_w / 2
        py = target_cy - new_h / 2
        final_x = px + lx
        final_y = py + ly
        new_kps.append((final_x, final_y, kp_vis))
    return new_kps


def make_yolo_label(bbox, kps, img_w, img_h):
    """Create a YOLO pose label string from pixel bbox and keypoints.
    Returns None if any visible keypoint is out of bounds."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # Check bbox in bounds
    for v in (cx, cy, w, h):
        if v < 0.0 or v > 1.0:
            return None
    # Check visible keypoints in bounds
    for kp_x, kp_y, kp_vis in kps:
        if kp_vis > 0:
            nx, ny = kp_x / img_w, kp_y / img_h
            if nx < 0.0 or nx > 1.0 or ny < 0.0 or ny > 1.0:
                return None
    parts = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
    for kp_x, kp_y, kp_vis in kps:
        parts.append(f"{kp_x / img_w:.6f} {kp_y / img_h:.6f} {kp_vis:.6f}")
    return " ".join(parts)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading SAM3 model...")
    sam_model = SAM(SAM3_PT)

    print("Finding empty board images...")
    all_empty_boards = find_images_by_dart_count(SRC_IMAGES_DIR, SRC_LABELS_DIR, 0)
    print(f"  Found {len(all_empty_boards)} empty board images")

    # ── Interactively select which boards to use as backgrounds ───────────
    print("\nSelect background images (A=accept, S=skip, Q=quit selection):")
    empty_boards = []
    for idx, (img_path, label_path, lines) in enumerate(all_empty_boards):
        img = cv2.imread(img_path)
        if img is None:
            continue
        basename = os.path.basename(img_path)
        win_name = f"Background {idx+1}/{len(all_empty_boards)} - {basename}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 900, 700)
        vis = img.copy()
        cv2.putText(vis, "A: accept  S: skip  Q: done selecting",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, "A: accept  S: skip  Q: done selecting",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow(win_name, vis)
        done = False
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('a'):
                empty_boards.append((img_path, label_path, lines))
                print(f"  [{idx+1}/{len(all_empty_boards)}] {basename} - ACCEPTED ({len(empty_boards)} selected)")
                break
            elif key == ord('s'):
                print(f"  [{idx+1}/{len(all_empty_boards)}] {basename} - skipped")
                break
            elif key == ord('q'):
                print(f"  Done selecting backgrounds.")
                done = True
                break
        cv2.destroyWindow(win_name)
        if done:
            break

    print(f"  Selected {len(empty_boards)} background images")

    print("Finding 1-dart images...")
    one_dart_images = find_images_by_dart_count(SRC_IMAGES_DIR, SRC_LABELS_DIR, 1)
    print(f"  Found {len(one_dart_images)} single-dart images")

    if not empty_boards:
        print("ERROR: No empty board images found!")
        return
    if not one_dart_images:
        print("ERROR: No single-dart images found!")
        return

    # ── Pre-extract all darts using SAM3 ────────────────────────────────────
    print("\nExtracting darts with SAM3...")
    extracted_darts = []  # list of (dart_rgba, crop_offset, keypoints)

    print("  Interactive mode: review each segmentation in the popup window.")
    print("  LMB=include point, RMB=exclude point, R=reset, A=accept, S=skip, Q=quit\n")

    for idx, (img_path, label_path, lines) in enumerate(one_dart_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        basename = os.path.basename(img_path)
        class_id, bbox, kps = parse_yolo_label(lines[0], w, h)

        print(f"  [{idx+1}/{len(one_dart_images)}] {basename}")
        result = interactive_segment(sam_model, img, bbox, basename)

        if isinstance(result, str) and result == 'quit':
            print("  User quit early.")
            break
        if result is None:
            print(f"    Skipped")
            continue

        mask = result
        dart_rgba, offset = extract_dart(img, mask, bbox)
        extracted_darts.append((dart_rgba, offset, kps))
        print(f"    Accepted ({len(extracted_darts)} darts total)")

    print(f"\nTotal extracted darts: {len(extracted_darts)}")

    # ── Generate composite images ───────────────────────────────────────────
    # Create output directories
    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUT_LABELS_DIR, exist_ok=True)

    print(f"\nGenerating composite images ({IMAGES_PER_BOARD} per board)...")
    print(f"Output directory: {OUT_DIR}")
    total_generated = 0

    for board_img_path, _, _ in empty_boards:
        board_img = cv2.imread(board_img_path)
        if board_img is None:
            continue
        h_board, w_board = board_img.shape[:2]
        board_basename = os.path.splitext(os.path.basename(board_img_path))[0]

        for gen_idx in range(IMAGES_PER_BOARD):
            composite = board_img.copy()
            n_darts = random.randint(MIN_DARTS, MAX_DARTS)
            dart_choices = random.sample(extracted_darts, min(n_darts, len(extracted_darts)))

            label_lines = []
            for dart_rgba, crop_offset, orig_kps in dart_choices:
                scale = random.uniform(SCALE_MIN, SCALE_MAX)

                # Random placement within central region of the board
                margin_x = int(w_board * 0.15)
                margin_y = int(h_board * 0.15)
                target_cx = random.randint(margin_x, w_board - margin_x)
                target_cy = random.randint(margin_y, h_board - margin_y)

                composite, actual_bbox = paste_dart(composite, dart_rgba, target_cx, target_cy, scale)
                if actual_bbox is None:
                    continue

                new_kps = transform_keypoints(orig_kps, crop_offset, scale, target_cx, target_cy, dart_rgba.shape)
                label_line = make_yolo_label(actual_bbox, new_kps, w_board, h_board)
                if label_line is not None:
                    label_lines.append(label_line)

            if not label_lines:
                continue

            # Save
            out_name = f"{board_basename}_copypaste_{gen_idx:03d}"
            out_img_path = os.path.join(OUT_IMAGES_DIR, out_name + ".jpg")
            out_label_path = os.path.join(OUT_LABELS_DIR, out_name + ".txt")
            cv2.imwrite(out_img_path, composite)
            with open(out_label_path, "w") as f:
                f.write("\n".join(label_lines) + "\n")
            total_generated += 1

    print(f"\nDone! Generated {total_generated} new images in {OUT_DIR}")


if __name__ == "__main__":
    main()
