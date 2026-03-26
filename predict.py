"""
OpenCV GUI application for dart pose prediction.

Usage:
    python predict.py
    python predict.py --model runs/pose/dart_pose6/weights/best.pt

Controls:
    M                     - Select a YOLO model (.pt / .onnx)
    O                     - Open image(s) for prediction
    Left/Right arrows     - Navigate between images
    Confidence slider     - Adjust confidence threshold
    Mouse wheel           - Zoom in/out (centered on cursor)
    Right-click drag      - Pan when zoomed in
    R                     - Reset zoom
    Del                   - Delete current image and its label file
    Q / Esc               - Quit
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs", "pose", "full-dataset-with-bias-lern", "weights", "best.pt"
)

WINDOW = "Dart Pose Detector"
KEYPOINT_NAMES = ["tip", "base"]
KEYPOINT_COLORS = [(0, 255, 0), (255, 0, 0)]  # green=tip, blue=base
LINE_COLOR = (0, 0, 255)
GT_TIP_COLOR = (0, 200, 0)      # darker green for ground truth tip
GT_BASE_COLOR = (200, 0, 0)     # darker blue for ground truth base
GT_LINE_COLOR = (200, 0, 200)   # magenta for ground truth line


def find_label_path(image_path):
    """Find a YOLO label .txt file for the given image.
    Checks: same dir, or parallel labels/ dir (images/x -> labels/x).
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.dirname(image_path)

    # Same directory
    candidate = os.path.join(img_dir, stem + ".txt")
    if os.path.isfile(candidate):
        return candidate

    # Parallel labels/ directory (e.g. images/train -> labels/train)
    parts = img_dir.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "images":
            label_dir = "/".join(parts[:i] + ["labels"] + parts[i + 1:])
            candidate = os.path.join(label_dir, stem + ".txt")
            if os.path.isfile(candidate):
                return candidate

    return None


def parse_yolo_labels(label_path, img_w, img_h):
    """Parse YOLO pose label file. Returns list of [(tip_x, tip_y), (base_x, base_y)] per dart."""
    results = []
    with open(label_path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 11:
                continue
            vals = [float(v) for v in vals]
            # vals: cls cx cy w h kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis
            tip = (vals[5] * img_w, vals[6] * img_h, vals[7])
            base = (vals[8] * img_w, vals[9] * img_h, vals[10])
            results.append((tip, base))
    return results


def _tk_open_files(title="Select files", filetypes=None):
    """Open a file dialog using tkinter. Returns list of paths or []."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(title=title, filetypes=filetypes or [("All files", "*.*")])
    root.destroy()
    return list(paths)


def _tk_open_file(title="Select file", filetypes=None):
    """Open a single-file dialog using tkinter. Returns path or None."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes or [("All files", "*.*")])
    root.destroy()
    return path or None


class App:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.images = []
        self.index = 0
        self.conf = 0.25
        self.tip_conf = 0.25
        self.display_img = None
        self.zoom = 1.0
        self.pan_x = 0.0  # pan offset in full-res image coords
        self.pan_y = 0.0
        self._dragging = False
        self._drag_start = None
        self._base_img = None  # full-res annotated image before zoom/crop

    def select_model(self):
        """Open file dialog to select a YOLO model."""
        path = _tk_open_file(
            title="Select YOLO model",
            filetypes=[("YOLO models", "*.pt *.onnx"), ("All files", "*.*")]
        )
        if path and os.path.isfile(path):
            self.model_path = path
            self.model = YOLO(path)
            self.predict_and_draw()

    def open_files(self):
        """Open file dialog to select images."""
        paths = _tk_open_files(
            title="Select images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All files", "*.*")]
        )
        if paths:
            self.images = sorted(paths)
            self.index = 0
            self.predict_and_draw()

    def predict_and_draw(self):
        """Run prediction on current image and update display."""
        if not self.images:
            self.display_img = self._blank("No image loaded. Press 'O' to open.")
            return

        path = self.images[self.index]
        img = cv2.imread(path)
        if img is None:
            self.display_img = self._blank(f"Failed to load: {path}")
            return

        results = self.model(img, conf=self.conf, iou=0.3, verbose=False)
        num_dets = 0

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints

            for i in range(len(boxes)):
                if keypoints is None:
                    continue
                kpts = keypoints.data[i]

                # Filter by tip keypoint confidence
                tip_conf = kpts[0][2].item() if len(kpts) > 0 else 0
                if tip_conf < self.tip_conf:
                    continue
                num_dets += 1

                for j, (name, color) in enumerate(zip(KEYPOINT_NAMES, KEYPOINT_COLORS)):
                    if j < len(kpts):
                        kx, ky, kc = kpts[j].tolist()
                        if kc > 0.3:
                            cv2.circle(img, (int(kx), int(ky)), 6, color, -1)
                            cv2.putText(img, f"{name} {kc:.2f}",
                                        (int(kx) + 8, int(ky) - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if len(kpts) >= 2:
                    tip = kpts[0].tolist()
                    base = kpts[1].tolist()
                    if tip[2] > 0.3 and base[2] > 0.3:
                        cv2.line(img, (int(tip[0]), int(tip[1])),
                                 (int(base[0]), int(base[1])), LINE_COLOR, 2)

        # Draw ground truth labels if available
        h, w = img.shape[:2]
        label_path = find_label_path(path)
        num_gt = 0
        if label_path:
            gt_darts = parse_yolo_labels(label_path, w, h)
            num_gt = len(gt_darts)
            for tip, base in gt_darts:
                if tip[2] > 0:
                    cv2.circle(img, (int(tip[0]), int(tip[1])), 8, GT_TIP_COLOR, 2)
                if base[2] > 0:
                    cv2.circle(img, (int(base[0]), int(base[1])), 8, GT_BASE_COLOR, 2)
                if tip[2] > 0 and base[2] > 0:
                    cv2.line(img, (int(tip[0]), int(tip[1])),
                             (int(base[0]), int(base[1])), GT_LINE_COLOR, 1)

        # Draw HUD bar at the top
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        model_name = os.path.basename(self.model_path)
        filename = os.path.basename(path)
        nav_text = f"[{self.index + 1}/{len(self.images)}] {filename}"
        gt_text = f" | {num_gt} GT" if num_gt else ""
        det_text = f"{num_dets} det{gt_text} | conf>={self.conf:.2f} tip>={self.tip_conf:.2f} | {model_name}"
        help_text = "M:Model O:Open </>:Nav Wheel:Zoom R:Reset Q:Quit"

        cv2.putText(img, nav_text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img, det_text, (w // 3, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(img, help_text, (w - 320, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        self._base_img = img
        self._apply_zoom()

    def _apply_zoom(self):
        """Crop and resize _base_img according to current zoom and pan, then set display_img."""
        if self._base_img is None:
            return
        img = self._base_img
        h, w = img.shape[:2]

        # Compute the fixed display size (same as zoom=1 fit-to-screen)
        max_dim = 1000
        if max(h, w) > max_dim:
            fit_scale = max_dim / max(h, w)
        else:
            fit_scale = 1.0
        disp_w = int(w * fit_scale)
        disp_h = int(h * fit_scale)

        # Viewport size in source pixels
        vw = w / self.zoom
        vh = h / self.zoom

        # Clamp pan so viewport stays within image
        self.pan_x = max(0.0, min(self.pan_x, w - vw))
        self.pan_y = max(0.0, min(self.pan_y, h - vh))

        x1, y1 = int(self.pan_x), int(self.pan_y)
        x2, y2 = int(self.pan_x + vw), int(self.pan_y + vh)
        crop = img[y1:y2, x1:x2]

        # Always resize to the fixed display size so the window stays constant
        crop = cv2.resize(crop, (disp_w, disp_h))

        self.display_img = crop

    def _on_mouse(self, event, x, y, flags, param):
        """Handle mouse events for zoom and pan."""
        if event == cv2.EVENT_MOUSEWHEEL:
            if self._base_img is None:
                return
            # Determine zoom direction
            if flags > 0:
                new_zoom = min(self.zoom * 1.2, 20.0)
            else:
                new_zoom = max(self.zoom / 1.2, 1.0)

            if new_zoom == self.zoom:
                return

            # Convert mouse position to source image coordinates
            disp_h, disp_w = self.display_img.shape[:2]
            bh, bw = self._base_img.shape[:2]
            vw_old = bw / self.zoom
            vh_old = bh / self.zoom
            # mouse fraction within display
            fx = x / disp_w if disp_w else 0.5
            fy = y / disp_h if disp_h else 0.5
            # source coords under cursor
            src_x = self.pan_x + fx * vw_old
            src_y = self.pan_y + fy * vh_old

            self.zoom = new_zoom
            vw_new = bw / self.zoom
            vh_new = bh / self.zoom
            # Re-center so the same source point stays under the cursor
            self.pan_x = src_x - fx * vw_new
            self.pan_y = src_y - fy * vh_new

            self._apply_zoom()
            if self.display_img is not None:
                cv2.imshow(WINDOW, self.display_img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._dragging = True
            self._drag_start = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            if self._drag_start and self._base_img is not None and self.display_img is not None:
                dx = x - self._drag_start[0]
                dy = y - self._drag_start[1]
                self._drag_start = (x, y)
                # Convert display pixel delta to source pixel delta
                disp_h, disp_w = self.display_img.shape[:2]
                bh, bw = self._base_img.shape[:2]
                vw = bw / self.zoom
                vh = bh / self.zoom
                self.pan_x -= dx * (vw / disp_w)
                self.pan_y -= dy * (vh / disp_h)
                self._apply_zoom()
                if self.display_img is not None:
                    cv2.imshow(WINDOW, self.display_img)

        elif event == cv2.EVENT_RBUTTONUP:
            self._dragging = False
            self._drag_start = None

    def on_conf_change(self, val):
        """Trackbar callback for box confidence threshold."""
        self.conf = val / 100.0
        self.predict_and_draw()
        if self.display_img is not None:
            cv2.imshow(WINDOW, self.display_img)

    def on_tip_conf_change(self, val):
        """Trackbar callback for tip keypoint confidence threshold."""
        self.tip_conf = val / 100.0
        self.predict_and_draw()
        if self.display_img is not None:
            cv2.imshow(WINDOW, self.display_img)

    @staticmethod
    def _blank(text):
        """Create a blank image with centered text."""
        img = np.zeros((400, 700, 3), dtype=np.uint8)
        cv2.putText(img, text, (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return img

    def delete_current(self):
        """Delete the current image and its associated label file."""
        if not self.images:
            return
        path = self.images[self.index]
        label_path = find_label_path(path)
        # Delete files
        deleted = []
        if os.path.isfile(path):
            os.remove(path)
            deleted.append(os.path.basename(path))
        if label_path and os.path.isfile(label_path):
            os.remove(label_path)
            deleted.append(os.path.basename(label_path))
        print(f"Deleted: {', '.join(deleted)}")
        # Update image list
        self.images.pop(self.index)
        if not self.images:
            self.display_img = self._blank("No images left. Press 'O' to open.")
            return
        self.index = min(self.index, len(self.images) - 1)
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.predict_and_draw()

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Box Conf", WINDOW, int(self.conf * 100), 100, self.on_conf_change)
        cv2.createTrackbar("Tip Conf", WINDOW, int(self.tip_conf * 100), 100, self.on_tip_conf_change)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

        self.display_img = self._blank("Press 'M' to select model, 'O' to open images.")
        cv2.imshow(WINDOW, self.display_img)

        while True:
            key = cv2.waitKeyEx(50)

            if key == ord('q') or key == 27:
                break
            elif key == ord('m'):
                self.select_model()
            elif key == ord('o'):
                self.open_files()
            elif key in (0x270000, 83, ord('d')):  # Right arrow (Win/Linux) or D
                if self.images:
                    self.index = (self.index + 1) % len(self.images)
                    self.zoom = 1.0
                    self.pan_x = self.pan_y = 0.0
                    self.predict_and_draw()
            elif key in (0x250000, 81, ord('a')):  # Left arrow (Win/Linux) or A
                if self.images:
                    self.index = (self.index - 1) % len(self.images)
                    self.zoom = 1.0
                    self.pan_x = self.pan_y = 0.0
                    self.predict_and_draw()
            elif key == ord('r'):
                self.zoom = 1.0
                self.pan_x = 0.0
                self.pan_y = 0.0
                self._apply_zoom()
            elif key == 0x2E0000 or key == 65535:  # Del key (Win extended / Linux)
                self.delete_current()

            if self.display_img is not None:
                cv2.imshow(WINDOW, self.display_img)

        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dart pose prediction GUI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model weights")
    args = parser.parse_args()

    app = App(args.model)
    app.run()


if __name__ == "__main__":
    main()
