"""
Homography Viewer — 3-Camera Dartboard Mapping
================================================
Click a point on any camera view to map it onto the flat dartboard image
via the board-plane homography computed from calibration_labels.json.

Windows: top_left, top_right, bottom_right, Dartboard

Controls:
    Left-click on any camera  — map point to dartboard
    c                         — clear all markers
    n                         — cycle to next timestamp's images
    p                         — cycle to previous timestamp's images
    q                         — quit
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
LABELS_PATH = SCRIPT_DIR / "calibration_labels.json"
BOARD_IMG_PATH = SCRIPT_DIR / "assets" / "dartboard.jpg"
SCREENSHOT_DIR = SCRIPT_DIR / "screenshots"

# ─── Camera names ─────────────────────────────────────────────────────────────
CAMERAS = ["top_left", "top_right", "bottom_right"]

# ─── Dartboard image mapping ─────────────────────────────────────────────────
# dartboard.jpg is 1024x1024, face-on view with 20 at top.
# World coords: origin = bullseye, X = right, Z axis with -Z = up on board.
# So the mapping is: u = cx + scale * X,  v = cy + scale * Z
BOARD_CENTER = (512, 512)
# Outer double ring (170 mm) appears at ~435 px radius in the image
BOARD_SCALE = 435.0 / 170.0  # px per mm ≈ 2.56

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
RED = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# One color per camera
CAM_COLORS = {
    "top_left": (0, 200, 255),    # orange
    "top_right": (255, 200, 0),   # cyan-ish
    "bottom_right": (200, 0, 255),  # pink
}

WIN_BOARD = "Dartboard"


# ─── Geometry ─────────────────────────────────────────────────────────────────

def world_xz_to_board_pixel(x: float, z: float) -> tuple[int, int]:
    """Map world board-plane (X, Z) in mm to dartboard.jpg pixel."""
    u = int(round(BOARD_CENTER[0] + BOARD_SCALE * x))
    v = int(round(BOARD_CENTER[1] + BOARD_SCALE * z))
    return u, v


def board_pixel_to_world_xz(u: int, v: int) -> tuple[float, float]:
    """Map dartboard.jpg pixel to world board-plane (X, Z) in mm."""
    x = (u - BOARD_CENTER[0]) / BOARD_SCALE
    z = (v - BOARD_CENTER[1]) / BOARD_SCALE
    return x, z


def compute_homography(labels: list[dict]) -> np.ndarray:
    """Compute homography from world (X, Z) → camera pixel.

    labels: list of {'pixel': [u,v], 'world': [X, 0, Z]}
    Returns H (3x3) such that pixel ~ H @ [X, Z, 1]^T.
    """
    src = np.array([[l["world"][0], l["world"][2]] for l in labels], dtype=np.float64)
    dst = np.array([l["pixel"] for l in labels], dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    # Reprojection error
    n = src.shape[0]
    ones = np.ones((n, 1))
    src_h = np.hstack([src, ones])
    proj = (H @ src_h.T).T
    proj_2d = proj[:, :2] / proj[:, 2:3]
    errors = np.linalg.norm(proj_2d - dst, axis=1)
    inliers = mask.ravel().sum() if mask is not None else n
    print(f"  Reprojection: mean={errors.mean():.2f}px, max={errors.max():.2f}px, inliers={inliers}/{n}")
    return H


# ─── Viewer ───────────────────────────────────────────────────────────────────

class HomographyViewer:
    def __init__(self):
        # Load labels
        if not LABELS_PATH.exists():
            sys.exit(f"Error: {LABELS_PATH} not found. Run calibrate_cameras.py first.")

        with open(LABELS_PATH) as f:
            all_labels = json.load(f)

        # Load dartboard image
        self.board_orig = cv2.imread(str(BOARD_IMG_PATH))
        if self.board_orig is None:
            sys.exit(f"Cannot load {BOARD_IMG_PATH}")

        # Compute homographies per camera
        self.H = {}       # world (X,Z) → camera pixel
        self.H_inv = {}   # camera pixel → world (X,Z)
        for cam in CAMERAS:
            if cam not in all_labels:
                sys.exit(f"No labels for camera '{cam}' in {LABELS_PATH}")
            print(f"[{cam}]")
            labels = all_labels[cam]["points"]
            self.H[cam] = compute_homography(labels)
            self.H_inv[cam] = np.linalg.inv(self.H[cam])

        # Gather all timestamps
        self.timestamps = self._find_timestamps()
        if not self.timestamps:
            sys.exit("No screenshot timestamps found.")
        self.ts_idx = 0
        print(f"\n{len(self.timestamps)} timestamps found. Use n/p to cycle.\n")

        # Load camera images for current timestamp
        self.cam_orig = {}
        self.cam_img = {}
        self._load_images()

        self.board_img = self.board_orig.copy()

        # Mapped points: list of (board_u, board_v, cam_name)
        self.mapped_points: list[tuple[int, int, str]] = []

    def _find_timestamps(self) -> list[str]:
        """Find all unique timestamps in the screenshots dir."""
        ts_set = set()
        for f in SCREENSHOT_DIR.iterdir():
            if f.suffix == ".png":
                # Format: YYYYMMDD_HHMMSS_camname.png
                parts = f.stem.split("_", 2)
                if len(parts) >= 3:
                    ts_set.add(f"{parts[0]}_{parts[1]}")
        return sorted(ts_set)

    def _load_images(self):
        ts = self.timestamps[self.ts_idx]
        for cam in CAMERAS:
            path = SCREENSHOT_DIR / f"{ts}_{cam}.png"
            img = cv2.imread(str(path))
            if img is None:
                print(f"Warning: cannot load {path}")
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
            self.cam_orig[cam] = img
            self.cam_img[cam] = img.copy()

    def _cam_pixel_to_board_xz(self, cam: str, u: int, v: int) -> tuple[float, float]:
        """Map camera pixel to world (X, Z) via inverse homography."""
        p = np.array([u, v, 1.0])
        xz = self.H_inv[cam] @ p
        xz /= xz[2]
        return xz[0], xz[1]

    def _world_xz_to_cam_pixel(self, cam: str, x: float, z: float) -> tuple[int, int]:
        """Map world (X, Z) to camera pixel via homography."""
        p = self.H[cam] @ np.array([x, z, 1.0])
        p /= p[2]
        return int(round(p[0])), int(round(p[1]))

    # ── Drawing ───────────────────────────────────────────────────────────

    def _redraw(self):
        # Reset images
        for cam in CAMERAS:
            self.cam_img[cam] = self.cam_orig[cam].copy()
        self.board_img = self.board_orig.copy()

        # Draw mapped points
        for bu, bv, source_cam in self.mapped_points:
            color = CAM_COLORS[source_cam]
            h_b, w_b = self.board_img.shape[:2]

            # On dartboard
            if 0 <= bu < w_b and 0 <= bv < h_b:
                cv2.drawMarker(self.board_img, (bu, bv), color, cv2.MARKER_CROSS, 25, 2)
                cv2.circle(self.board_img, (bu, bv), 10, color, 2)

            # Project to all camera views
            x, z = board_pixel_to_world_xz(bu, bv)
            for cam in CAMERAS:
                cu, cv_ = self._world_xz_to_cam_pixel(cam, x, z)
                h_c, w_c = self.cam_img[cam].shape[:2]
                if 0 <= cu < w_c and 0 <= cv_ < h_c:
                    cv2.circle(self.cam_img[cam], (cu, cv_), 6, color, -1)
                    cv2.circle(self.cam_img[cam], (cu, cv_), 8, BLACK, 1)

        # Status bars
        ts = self.timestamps[self.ts_idx]
        for cam in CAMERAS:
            img = self.cam_img[cam]
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w, 30), BLACK, -1)
            status = f"{cam} | {ts} ({self.ts_idx+1}/{len(self.timestamps)}) | click=map | n/p=nav | c=clear | q=quit"
            cv2.putText(img, status, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        # Board status
        h_b, w_b = self.board_img.shape[:2]
        cv2.rectangle(self.board_img, (0, 0), (w_b, 30), BLACK, -1)
        cv2.putText(self.board_img, f"Dartboard | {len(self.mapped_points)} points mapped", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        # Show all
        for cam in CAMERAS:
            cv2.imshow(cam, self.cam_img[cam])
        cv2.imshow(WIN_BOARD, self.board_img)

    # ── Mouse callbacks ───────────────────────────────────────────────────

    def _make_click_handler(self, cam_name: str):
        def handler(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            # Map camera pixel → board plane → dartboard image
            bx, bz = self._cam_pixel_to_board_xz(cam_name, x, y)
            bu, bv = world_xz_to_board_pixel(bx, bz)

            # Distance from center in mm
            dist_mm = np.sqrt(bx**2 + bz**2)

            print(f"  [{cam_name}] ({x}, {y}) -> board ({bx:.1f}, {bz:.1f}) mm "
                  f"-> dartboard ({bu}, {bv})  dist={dist_mm:.1f}mm")

            self.mapped_points.append((bu, bv, cam_name))
            self._redraw()

        return handler

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        # Create windows
        for cam in CAMERAS:
            cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam, 960, 540)
            cv2.setMouseCallback(cam, self._make_click_handler(cam))

        cv2.namedWindow(WIN_BOARD, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_BOARD, 700, 700)

        self._redraw()

        print("Homography Viewer ready.")
        print("  Click on any camera to map the point to the dartboard.")
        print("  The point is also back-projected onto all other cameras.")
        print("  n/p = next/prev timestamp, c = clear, q = quit\n")

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                self.mapped_points.clear()
                self._redraw()
                print("Cleared.")

            elif key == ord('n'):
                self.ts_idx = (self.ts_idx + 1) % len(self.timestamps)
                self._load_images()
                self.mapped_points.clear()
                self._redraw()
                print(f"Timestamp: {self.timestamps[self.ts_idx]}")

            elif key == ord('p'):
                self.ts_idx = (self.ts_idx - 1) % len(self.timestamps)
                self._load_images()
                self.mapped_points.clear()
                self._redraw()
                print(f"Timestamp: {self.timestamps[self.ts_idx]}")

        cv2.destroyAllWindows()


def main():
    viewer = HomographyViewer()
    viewer.run()


if __name__ == "__main__":
    main()
