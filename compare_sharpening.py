"""
Compare image sharpening approaches on a video feed.

Shows one method at a time — swap between them with arrow keys.
Each frame shows the method name, latency, and RAM usage.

Usage:
    python compare_sharpening.py
    python compare_sharpening.py --video path/to/video.mp4
    python compare_sharpening.py --scale 0.5

Controls:
    Left/Right arrow  - Switch sharpening method
    Space             - Pause / Resume
    .                 - Step forward one frame (while paused)
    ,                 - Step backward one frame (while paused)
    Q/Esc             - Quit
"""

import os
import sys
import time
import cv2
import numpy as np
import psutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(PROJECT_DIR, "bottom_right.mp4")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
WINDOW = "Sharpening Comparison"


# ---------------------------------------------------------------------------
# Sharpening methods
# ---------------------------------------------------------------------------

class OriginalMethod:
    name = "Original"

    def process(self, frame):
        return frame.copy()


class UnsharpMaskMethod:
    name = "Unsharp Mask"

    def process(self, frame):
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)


class LaplacianSharpenMethod:
    name = "Laplacian Sharpen"

    def process(self, frame):
        lap = cv2.Laplacian(frame, cv2.CV_64F)
        sharpened = frame.astype(np.float64) + lap
        return np.clip(sharpened, 0, 255).astype(np.uint8)


class BilateralSharpenMethod:
    name = "Bilateral + Sharpen"

    def process(self, frame):
        smooth = cv2.bilateralFilter(frame, 9, 75, 75)
        return cv2.addWeighted(frame, 1.5, smooth, -0.5, 0)


class FSRCNNMethod:
    name = "FSRCNN x2"

    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "FSRCNN_x2.pb")
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        self.sr.setModel("fsrcnn", 2)

    def process(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        upscaled = self.sr.upsample(small)
        if upscaled.shape[:2] != (h, w):
            upscaled = cv2.resize(upscaled, (w, h))
        return upscaled


class RealESRGANMethod:
    name = "Real-ESRGAN x2"

    def __init__(self):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = os.path.join(MODELS_DIR, "RealESRGAN_x2plus.pth")
        rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=2)
        self.upsampler = RealESRGANer(
            scale=2, model_path=model_path, model=rrdb,
            tile=0, tile_pad=10, pre_pad=0, half=True,
        )

    def process(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        output, _ = self.upsampler.enhance(small, outscale=2)
        if output.shape[:2] != (h, w):
            output = cv2.resize(output, (w, h))
        return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_ram_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare sharpening methods")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to input video")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Display scale factor (e.g. 0.5 to halve panel size)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        sys.exit(1)

    print("Loading sharpening methods...")
    methods = [OriginalMethod(), UnsharpMaskMethod(), LaplacianSharpenMethod(),
               BilateralSharpenMethod()]

    try:
        methods.append(FSRCNNMethod())
        print("  FSRCNN loaded")
    except Exception as e:
        print(f"  FSRCNN unavailable: {e}")

    try:
        methods.append(RealESRGANMethod())
        print("  Real-ESRGAN loaded")
    except Exception as e:
        print(f"  Real-ESRGAN unavailable: {e}")

    current_idx = 0
    print(f"Loaded {len(methods)} methods. Left/Right to swap, Q to quit.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    paused = False
    last_frame = None
    last_latency = 0.0

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 83 or key == ord('d'):  # Right arrow or D
            current_idx = (current_idx + 1) % len(methods)
            print(f"  -> {methods[current_idx].name}")
            if paused and last_frame is not None:
                # Re-process current frame with new method
                t0 = time.perf_counter()
                result = methods[current_idx].process(last_frame)
                last_latency = (time.perf_counter() - t0) * 1000
                _show(result, methods[current_idx], current_idx, len(methods),
                      last_latency, cap, args.scale)
                continue
        elif key == 81 or key == ord('a'):  # Left arrow or A
            current_idx = (current_idx - 1) % len(methods)
            print(f"  -> {methods[current_idx].name}")
            if paused and last_frame is not None:
                t0 = time.perf_counter()
                result = methods[current_idx].process(last_frame)
                last_latency = (time.perf_counter() - t0) * 1000
                _show(result, methods[current_idx], current_idx, len(methods),
                      last_latency, cap, args.scale)
                continue
        elif key == ord('.') and paused:
            pass  # fall through to read next frame
        elif key == ord(',') and paused:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
        elif paused:
            continue

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        last_frame = frame
        method = methods[current_idx]

        t0 = time.perf_counter()
        result = method.process(frame)
        last_latency = (time.perf_counter() - t0) * 1000

        _show(result, method, current_idx, len(methods),
              last_latency, cap, args.scale)

    cap.release()
    cv2.destroyAllWindows()


def _show(result, method, idx, total, latency_ms, cap, scale):
    h, w = result.shape[:2]
    display = cv2.resize(result, (int(w * scale), int(h * scale)))

    ram = get_ram_mb()
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Method name (top-left)
    label = f"[{idx + 1}/{total}] {method.name}"
    cv2.putText(display, label, (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Latency (below name)
    cv2.putText(display, f"Latency: {latency_ms:.1f} ms", (8, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # RAM + frame (bottom-left)
    info = f"RAM: {ram:.0f} MB | Frame: {frame_num}"
    cv2.putText(display, info, (8, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Controls hint (bottom-right)
    hint = "<-/-> swap | Space pause | Q quit"
    text_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(display, hint, (display.shape[1] - text_size[0] - 8, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow(WINDOW, display)


if __name__ == "__main__":
    main()
