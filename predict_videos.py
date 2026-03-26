"""
Run dart pose model on 3 videos simultaneously and display side-by-side.
Keypoint predictions are smoothed in real-time with a 1 Euro filter.

Usage:
    python predict_videos.py
    python predict_videos.py --model runs/pose/dart_pose_overfit_all/weights/best.pt
    python predict_videos.py --conf 0.5
    python predict_videos.py --min-cutoff 0.3 --beta 0.005

Controls:
    Space       - Pause / Resume
    . (period)  - Step forward one frame (while paused)
    , (comma)   - Step backward one frame (while paused)
    R           - Set current detections as reference
    M           - Manual label mode: click tip then base for each dart
                  (Left-click = place point, Right-click = undo, Enter = confirm, Esc = cancel)
    C           - Clear reference
    L           - Screenshot with reference labels overlaid
    S           - Screenshot each video (raw frame, no labels)
    Q/Esc       - Quit
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_DIR, "runs", "pose", "full-dataset-with-bias-lern", "weights", "best.pt")

VIDEOS = [
    # os.path.join(PROJECT_DIR, "top_left.mp4"),
    # os.path.join(PROJECT_DIR, "top_right.mp4"),
    os.path.join(PROJECT_DIR, "bottom_right.mp4"),
]

KEYPOINT_NAMES = ["tip", "base"]
KEYPOINT_COLORS = [(0, 255, 0), (255, 0, 0)]  # green=tip, blue=base
LINE_COLOR = (0, 0, 255)
REF_TIP_COLOR = (0, 200, 200)      # yellow-ish for reference tip
REF_BASE_COLOR = (200, 200, 0)     # cyan-ish for reference base
REF_LINE_COLOR = (0, 165, 255)     # orange for reference line
WINDOW = "Dart Pose - 3 Videos"


class OneEuroFilter:
    """1 Euro filter for smoothing a single scalar signal.

    When the signal is nearly static (low velocity), cutoff stays at min_cutoff,
    giving heavy smoothing. When the signal moves fast, beta increases the cutoff
    to reduce lag. Lower min_cutoff = smoother when still.
    """

    def __init__(self, min_cutoff=0.3, beta=0.005, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class KeypointSmoother:
    """Applies 1 Euro filters per detection slot per video stream.

    Each slot tracks one dart's tip_x, tip_y, base_x, base_y independently.
    """

    def __init__(self, max_darts=5, min_cutoff=0.3, beta=0.005):
        self.max_darts = max_darts
        # filters[dart_idx] = (tip_x, tip_y, base_x, base_y)
        self.filters = [
            [OneEuroFilter(min_cutoff, beta) for _ in range(4)]
            for _ in range(max_darts)
        ]

    def smooth(self, t, keypoints_data):
        """Smooth keypoints tensor (N, 2, 3). Returns a new smoothed clone."""
        data = keypoints_data.clone()
        for i in range(min(len(data), self.max_darts)):
            kpts = data[i]
            tip_x, tip_y, tip_c = kpts[0].tolist()
            base_x, base_y, base_c = kpts[1].tolist()
            fx_tip_x, fx_tip_y, fx_base_x, fx_base_y = self.filters[i]

            if tip_c > 0.3:
                data[i][0][0] = fx_tip_x(t, tip_x)
                data[i][0][1] = fx_tip_y(t, tip_y)
            if base_c > 0.3:
                data[i][1][0] = fx_base_x(t, base_x)
                data[i][1][1] = fx_base_y(t, base_y)
        return data


def extract_keypoints_from_results(results):
    """Extract detections as a list of dicts with box and keypoints per detection.

    Each entry: {
        'box': (cx, cy, w, h) in pixels,
        'tip': (x, y, conf),
        'base': (x, y, conf),
    }
    """
    darts = []
    for result in results:
        if result.keypoints is None:
            continue
        for i in range(len(result.boxes)):
            kpts = result.keypoints.data[i]
            if len(kpts) >= 2:
                tip = kpts[0].tolist()
                base = kpts[1].tolist()
                # xywh: center-x, center-y, width, height in pixels
                box = result.boxes.xywh[i].tolist()
                darts.append({'box': box, 'tip': tip, 'base': base})
    return darts


def draw_predictions(frame, results):
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        for i in range(len(boxes)):
            if keypoints is not None:
                kpts = keypoints.data[i]
                for j, (name, color) in enumerate(zip(KEYPOINT_NAMES, KEYPOINT_COLORS)):
                    if j < len(kpts):
                        kx, ky, kc = kpts[j].tolist()
                        if kc > 0.3:
                            cv2.circle(frame, (int(kx), int(ky)), 6, color, -1)
                            cv2.putText(frame, f"{name} {kc:.2f}",
                                        (int(kx) + 8, int(ky) - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                if len(kpts) >= 2:
                    tip = kpts[0].tolist()
                    base = kpts[1].tolist()
                    if tip[2] > 0.3 and base[2] > 0.3:
                        cv2.line(frame, (int(tip[0]), int(tip[1])),
                                 (int(base[0]), int(base[1])), LINE_COLOR, 2)
    return frame


def draw_reference(frame, ref_darts):
    """Draw reference keypoints on a frame."""
    for dart_idx, dart in enumerate(ref_darts):
        tip, base = dart['tip'], dart['base']
        if tip[2] > 0.3:
            cv2.circle(frame, (int(tip[0]), int(tip[1])), 8, REF_TIP_COLOR, 2)
            cv2.putText(frame, f"ref{dart_idx+1}:tip",
                        (int(tip[0]) + 10, int(tip[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, REF_TIP_COLOR, 1)
        if base[2] > 0.3:
            cv2.circle(frame, (int(base[0]), int(base[1])), 8, REF_BASE_COLOR, 2)
            cv2.putText(frame, f"ref{dart_idx+1}:base",
                        (int(base[0]) + 10, int(base[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, REF_BASE_COLOR, 1)
        if tip[2] > 0.3 and base[2] > 0.3:
            cv2.line(frame, (int(tip[0]), int(tip[1])),
                     (int(base[0]), int(base[1])), REF_LINE_COLOR, 2)


MANUAL_TIP_COLOR = (0, 255, 0)       # green for manual tip
MANUAL_BASE_COLOR = (255, 0, 0)      # blue for manual base
MANUAL_LINE_COLOR = (0, 200, 255)    # orange for manual dart line
MANUAL_PENDING_COLOR = (0, 255, 255) # yellow for next-expected point


class ManualLabeler:
    """State machine for manually clicking tip/base pairs on a single video panel."""

    def __init__(self):
        self.active = False
        self.vid_idx = 0          # which video panel is being labeled
        self.points = []          # flat list: [tip1, base1, tip2, base2, ...]
        self.panel_offsets = []   # x-offset of each panel in the combined image
        self.panel_widths = []    # width of each panel
        self.scale = 1.0          # display-to-original scale factor

    def start(self, vid_idx, panel_offsets, panel_widths, scale):
        self.active = True
        self.vid_idx = vid_idx
        self.points = []
        self.panel_offsets = panel_offsets
        self.panel_widths = panel_widths
        self.scale = scale

    def cancel(self):
        self.active = False
        self.points = []

    def handle_click(self, x, y, button):
        """Handle a mouse click. button: 'left' or 'right'."""
        if not self.active:
            return
        if button == 'right':
            if self.points:
                removed = self.points.pop()
                kind = "tip" if len(self.points) % 2 == 0 else "base"
                print(f"  Undo {kind} at ({removed[0]:.0f}, {removed[1]:.0f})")
            return
        # Left click: map display coords to original frame coords for the active panel
        offset = self.panel_offsets[self.vid_idx]
        pw = self.panel_widths[self.vid_idx]
        local_x = x - offset
        if local_x < 0 or local_x >= pw:
            return  # click outside the active panel
        orig_x = local_x / self.scale
        orig_y = y / self.scale
        self.points.append((orig_x, orig_y))
        kind = "tip" if len(self.points) % 2 == 1 else "base"
        dart_num = (len(self.points) + 1) // 2
        print(f"  Dart {dart_num} {kind} at ({orig_x:.0f}, {orig_y:.0f})")

    def get_darts(self):
        """Return completed dart dicts from the placed points."""
        darts = []
        for i in range(0, len(self.points) - 1, 2):
            tip = self.points[i]
            base = self.points[i + 1]
            # Build a bounding box from tip/base with some padding
            cx = (tip[0] + base[0]) / 2
            cy = (tip[1] + base[1]) / 2
            w = abs(tip[0] - base[0]) + 20
            h = abs(tip[1] - base[1]) + 20
            darts.append({
                'box': (cx, cy, w, h),
                'tip': [tip[0], tip[1], 1.0],
                'base': [base[0], base[1], 1.0],
            })
        return darts

    def draw_overlay(self, combined, panel_h):
        """Draw manual label points on the combined display image."""
        if not self.active:
            return
        offset = self.panel_offsets[self.vid_idx]
        for i, pt in enumerate(self.points):
            dx = int(pt[0] * self.scale) + offset
            dy = int(pt[1] * self.scale)
            is_tip = (i % 2 == 0)
            color = MANUAL_TIP_COLOR if is_tip else MANUAL_BASE_COLOR
            label = "tip" if is_tip else "base"
            dart_num = i // 2 + 1
            cv2.circle(combined, (dx, dy), 7, color, -1)
            cv2.putText(combined, f"d{dart_num}:{label}",
                        (dx + 10, dy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # Draw line from tip to base for completed pairs
            if not is_tip:
                tip_pt = self.points[i - 1]
                tx = int(tip_pt[0] * self.scale) + offset
                ty = int(tip_pt[1] * self.scale)
                cv2.line(combined, (tx, ty), (dx, dy), MANUAL_LINE_COLOR, 2)

        # HUD instructions
        n_complete = len(self.points) // 2
        next_kind = "TIP" if len(self.points) % 2 == 0 else "BASE"
        next_dart = len(self.points) // 2 + 1
        hud = (f"MANUAL LABEL: {n_complete} darts | "
               f"Click dart {next_dart} {next_kind} | "
               f"Right-click=undo | Enter=confirm | Esc=cancel")
        cv2.putText(combined, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def write_yolo_label(label_path, ref_darts, img_w, img_h):
    """Write reference darts as a YOLO pose label file.

    Format per line: cls cx cy w h kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis
    All values normalized to [0, 1].
    """
    with open(label_path, 'w') as f:
        for dart in ref_darts:
            cx, cy, bw, bh = dart['box']
            tip, base = dart['tip'], dart['base']
            # Normalize
            cx_n = cx / img_w
            cy_n = cy / img_h
            bw_n = bw / img_w
            bh_n = bh / img_h
            tip_x_n = tip[0] / img_w
            tip_y_n = tip[1] / img_h
            tip_v = 2 if tip[2] > 0.3 else 0
            base_x_n = base[0] / img_w
            base_y_n = base[1] / img_h
            base_v = 2 if base[2] > 0.3 else 0
            f.write(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f} "
                    f"{tip_x_n:.6f} {tip_y_n:.6f} {tip_v} "
                    f"{base_x_n:.6f} {base_y_n:.6f} {base_v}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run dart pose on 3 videos simultaneously")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--min-cutoff", type=float, default=0.001, help="1 Euro filter min cutoff (lower = smoother when static)")
    parser.add_argument("--beta", type=float, default=0.002, help="1 Euro filter beta (higher = less lag during motion)")
    args = parser.parse_args()

    model = YOLO(args.model)
    caps = [cv2.VideoCapture(v) for v in VIDEOS]

    for v, cap in zip(VIDEOS, caps):
        if not cap.isOpened():
            print(f"ERROR: Cannot open {v}")
            return

    smoothers = [KeypointSmoother(min_cutoff=args.min_cutoff, beta=args.beta) for _ in VIDEOS]

    panel_h = 480
    paused = False
    t_start = time.monotonic()

    screenshots_dir = os.path.join(PROJECT_DIR, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    raw_frames = []

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    prev_frames = [None] * len(VIDEOS)
    prev_results = [None] * len(VIDEOS)

    # Reference keypoints: list per video, each is a list of (tip, base) tuples
    ref_keypoints = [None] * len(VIDEOS)

    manual = ManualLabeler()

    def on_mouse(event, x, y, flags, param):
        if not manual.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            manual.handle_click(x, y, 'left')
        elif event == cv2.EVENT_RBUTTONDOWN:
            manual.handle_click(x, y, 'right')

    cv2.setMouseCallback(WINDOW, on_mouse)

    # Panel geometry for coordinate mapping (updated each frame)
    panel_offsets = [0]
    panel_widths = [640]
    last_scale = 1.0

    while True:
        step_forward = False
        step_backward = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or (key == 27 and not manual.active):
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("Paused. Use '.' to step forward, ',' to step backward.")
        elif key == ord('.') and paused:
            step_forward = True
        elif key == ord(',') and paused:
            step_backward = True
        elif key == ord('m') and not manual.active:
            # Enter manual label mode on the first (or only) video panel
            paused = True
            vid_idx = 0
            manual.start(vid_idx, panel_offsets, panel_widths, last_scale)
            print(f"Manual label mode: click TIP then BASE for each dart on panel {vid_idx}")
            print("  Right-click = undo | Enter = confirm | Esc = cancel")
        elif key == 13 and manual.active:  # Enter
            darts = manual.get_darts()
            if darts:
                vid_idx = manual.vid_idx
                if ref_keypoints[vid_idx]:
                    ref_keypoints[vid_idx].extend(darts)
                else:
                    ref_keypoints[vid_idx] = darts
                total = len(ref_keypoints[vid_idx])
                print(f"Manual reference added: {len(darts)} darts (total {total} on panel {vid_idx})")
            else:
                print("No complete darts to save (need tip+base pairs)")
            manual.cancel()
        elif key == 27 and manual.active:  # Esc cancels manual mode only
            manual.cancel()
            print("Manual label cancelled")
        elif key == ord('r') and not manual.active:
            # Save current detections as reference
            for vid_idx in range(len(VIDEOS)):
                if prev_results[vid_idx] is not None:
                    ref_keypoints[vid_idx] = extract_keypoints_from_results(prev_results[vid_idx])
            total = sum(len(r) for r in ref_keypoints if r)
            print(f"Reference set: {total} darts across {len(VIDEOS)} views")
        elif key == ord('c'):
            ref_keypoints = [None] * len(VIDEOS)
            print("Reference cleared")
        elif key == ord('l') and raw_frames:
            # Save raw frames as images + YOLO label txt files in screenshots-label-conf/
            label_conf_dir = os.path.join(PROJECT_DIR, "screenshots-label-conf")
            img_dir = os.path.join(label_conf_dir, "images")
            lbl_dir = os.path.join(label_conf_dir, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            names = ["top_left", "top_right", "bottom_right"]
            saved = 0
            for vid_idx, (name, frame) in enumerate(zip(names, raw_frames)):
                stem = f"{ts}_{name}"
                # Save raw image (no annotations)
                cv2.imwrite(os.path.join(img_dir, f"{stem}.png"), frame)
                # Write YOLO label from reference keypoints
                if ref_keypoints[vid_idx]:
                    h, w = frame.shape[:2]
                    write_yolo_label(os.path.join(lbl_dir, f"{stem}.txt"),
                                     ref_keypoints[vid_idx], w, h)
                    saved += len(ref_keypoints[vid_idx])
            print(f"Saved {saved} labels to {label_conf_dir}/ ({ts})")
        elif key == ord('s') and raw_frames:
            ts = time.strftime("%Y%m%d_%H%M%S")
            names = ["top_left", "top_right", "bottom_right"]
            for name, frame in zip(names, raw_frames):
                path = os.path.join(screenshots_dir, f"{ts}_{name}.png")
                cv2.imwrite(path, frame)
            print(f"Screenshots saved to {screenshots_dir}/{ts}_*.png")

        need_update = (not paused) or step_forward or step_backward

        if need_update:
            if step_backward:
                for cap in caps:
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))

            t = time.monotonic() - t_start
            raw_frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        raw_frames.append(np.zeros((panel_h, 640, 3), dtype=np.uint8))
                        continue
                raw_frames.append(frame)

            annotated = []
            for vid_idx, frame in enumerate(raw_frames):
                display = frame.copy()
                # Skip inference if frame is identical to previous (avoids GPU non-determinism jitter)
                if prev_frames[vid_idx] is not None and np.array_equal(frame, prev_frames[vid_idx]):
                    results = prev_results[vid_idx]
                else:
                    results = model.track(display, conf=args.conf, iou=args.iou, persist=True, tracker="bytetrack.yaml", verbose=False)
                    for result in results:
                        if result.keypoints is not None:
                            result.keypoints.data = smoothers[vid_idx].smooth(t, result.keypoints.data)
                    prev_results[vid_idx] = results
                prev_frames[vid_idx] = frame
                draw_predictions(display, results)
                # Draw reference overlay if set
                if ref_keypoints[vid_idx]:
                    draw_reference(display, ref_keypoints[vid_idx])
                annotated.append(display)

            panels = []
            panel_offsets = []
            panel_widths = []
            labels = ["Top Left", "Top Right", "Bottom Right"]
            x_offset = 0
            for panel, label in zip(annotated, labels):
                h, w = panel.shape[:2]
                scale = panel_h / h
                last_scale = scale
                resized = cv2.resize(panel, (int(w * scale), panel_h))
                cv2.putText(resized, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                panel_offsets.append(x_offset)
                panel_widths.append(resized.shape[1])
                x_offset += resized.shape[1]
                panels.append(resized)

            # Update manual labeler geometry
            if manual.active:
                manual.panel_offsets = panel_offsets
                manual.panel_widths = panel_widths
                manual.scale = last_scale

            # HUD bar
            combined = np.hstack(panels)
            has_ref = any(r is not None for r in ref_keypoints)
            status = "PAUSED" if paused else "PLAYING"
            hud = f"{status} | conf={args.conf} iou={args.iou}"
            if has_ref:
                hud += " | REF ACTIVE (C=clear)"
            else:
                hud += " | R=set ref | M=manual label"
            cv2.putText(combined, hud, (10, combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw manual labeling overlay
            if manual.active:
                manual.draw_overlay(combined, panel_h)

            cv2.imshow(WINDOW, combined)

            if step_forward or step_backward:
                paused = True

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
