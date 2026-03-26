"""
Lightweight dart pose prediction GUI — runs on Raspberry Pi.
Only requires: opencv-python, numpy (no ultralytics).

Usage:
    python predict_lite.py
    python predict_lite.py --model best.onnx

Controls:
    M                     - Select a YOLO model (.onnx)
    O                     - Open image(s) for prediction
    Left/Right arrows     - Navigate between images
    Confidence slider     - Adjust confidence threshold
    Q / Esc               - Quit
"""

import argparse
import os
import subprocess

import cv2
import numpy as np

WINDOW = "Dart Pose Detector"
KEYPOINT_NAMES = ["tip", "base"]
KEYPOINT_COLORS = [(0, 255, 0), (255, 0, 0)]  # green=tip, blue=base
LINE_COLOR = (0, 0, 255)
INPUT_SIZE = 640


def zenity_open_files(title="Select files", file_filter=None):
    """Open a multi-file dialog using zenity."""
    cmd = ["zenity", "--file-selection", "--multiple", "--separator=\n", f"--title={title}"]
    if file_filter:
        cmd.append(f"--file-filter={file_filter}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []


def zenity_open_file(title="Select file", file_filter=None):
    """Open a single-file dialog using zenity."""
    cmd = ["zenity", "--file-selection", f"--title={title}"]
    if file_filter:
        cmd.append(f"--file-filter={file_filter}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def preprocess(img, size=INPUT_SIZE):
    """Letterbox resize and normalize to NCHW float32 blob."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    pad_top = (size - nh) // 2
    pad_left = (size - nw) // 2

    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    blob = blob[np.newaxis]  # add batch dim
    return blob, scale, pad_left, pad_top


def postprocess(output, conf_thresh, scale, pad_left, pad_top):
    """Parse YOLO pose output [1, N, 12] -> list of (keypoints, conf).
    Each detection: [x1, y1, x2, y2, conf, cls, kp1_x, kp1_y, kp1_c, kp2_x, kp2_y, kp2_c]
    """
    dets = output[0]  # (N, 12)
    results = []
    for det in dets:
        c = det[4]
        if c < conf_thresh:
            continue
        # Unscale keypoints from letterboxed coords to original image coords
        kpts = []
        for k in range(2):
            kx = (det[6 + k * 3] - pad_left) / scale
            ky = (det[7 + k * 3] - pad_top) / scale
            kc = det[8 + k * 3]
            kpts.append((kx, ky, kc))
        results.append((kpts, c))
    return results


class App:
    def __init__(self, model_path):
        self.model_path = model_path
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.images = []
        self.index = 0
        self.conf = 0.25
        self.display_img = None

    def select_model(self):
        path = zenity_open_file(
            title="Select ONNX model",
            file_filter="ONNX models | *.onnx"
        )
        if path and os.path.isfile(path):
            self.model_path = path
            self.net = cv2.dnn.readNetFromONNX(path)
            self.predict_and_draw()

    def open_files(self):
        paths = zenity_open_files(
            title="Select images",
            file_filter="Images | *.jpg *.jpeg *.png *.bmp *.tiff *.webp"
        )
        if paths:
            self.images = sorted(paths)
            self.index = 0
            self.predict_and_draw()

    def predict_and_draw(self):
        if not self.images:
            self.display_img = _blank("No image loaded. Press 'O' to open.")
            return

        path = self.images[self.index]
        img = cv2.imread(path)
        if img is None:
            self.display_img = _blank(f"Failed to load: {path}")
            return

        blob, scale, pad_left, pad_top = preprocess(img)
        self.net.setInput(blob)
        output = self.net.forward()

        detections = postprocess(output, self.conf, scale, pad_left, pad_top)

        for kpts, conf in detections:
            for j, (name, color) in enumerate(zip(KEYPOINT_NAMES, KEYPOINT_COLORS)):
                kx, ky, kc = kpts[j]
                if kc > 0.3:
                    cv2.circle(img, (int(kx), int(ky)), 6, color, -1)
                    cv2.putText(img, f"{name} {kc:.2f}",
                                (int(kx) + 8, int(ky) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            tip, base = kpts[0], kpts[1]
            if tip[2] > 0.3 and base[2] > 0.3:
                cv2.line(img, (int(tip[0]), int(tip[1])),
                         (int(base[0]), int(base[1])), LINE_COLOR, 2)

        # HUD
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        model_name = os.path.basename(self.model_path)
        filename = os.path.basename(path)
        nav_text = f"[{self.index + 1}/{len(self.images)}] {filename}"
        det_text = f"{len(detections)} det | conf>={self.conf:.2f} | {model_name}"
        help_text = "M:Model O:Open </>:Nav Q:Quit"

        cv2.putText(img, nav_text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img, det_text, (w // 3, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(img, help_text, (w - 320, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        max_dim = 1000
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))

        self.display_img = img

    def on_conf_change(self, val):
        self.conf = val / 100.0
        self.predict_and_draw()
        if self.display_img is not None:
            cv2.imshow(WINDOW, self.display_img)

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Confidence", WINDOW, int(self.conf * 100), 100, self.on_conf_change)

        self.display_img = _blank("Press 'M' to select model, 'O' to open images.")
        cv2.imshow(WINDOW, self.display_img)

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('m'):
                self.select_model()
            elif key == ord('o'):
                self.open_files()
            elif key == 83 or key == ord('d'):
                if self.images:
                    self.index = (self.index + 1) % len(self.images)
                    self.predict_and_draw()
            elif key == 81 or key == ord('a'):
                if self.images:
                    self.index = (self.index - 1) % len(self.images)
                    self.predict_and_draw()

            if self.display_img is not None:
                cv2.imshow(WINDOW, self.display_img)

        cv2.destroyAllWindows()


def _blank(text):
    img = np.zeros((400, 700, 3), dtype=np.uint8)
    cv2.putText(img, text, (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Lightweight dart pose prediction GUI")
    parser.add_argument("--model", default="best.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    app = App(args.model)
    app.run()


if __name__ == "__main__":
    main()
