"""
Manual camera calibration using known dartboard geometry.

For each of the 3 cameras (top_left, top_right, bottom_right):
  1. Displays an image and prompts you to click known dartboard points in order.
  2. Computes the 3x4 projection matrix P via DLT.
  3. Decomposes P into intrinsic K and extrinsic [R|t].

Controls:
  Left-click  — mark the current point
  Right-click — skip the current point
  'u'         — undo last click
  'q'         — abort
  Enter       — finish labeling this camera (need >= 6 points)

Dartboard coordinate system:
  Origin = bullseye center, board lies in the XZ plane (Y = 0).
  X = right, Z = down (looking at the board face-on), Y = into the board.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ─── Standard dartboard dimensions (mm) ──────────────────────────────────────
# Number order clockwise starting from the top-right wire boundary of 20
NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Radii from center (mm)
BULL_RADIUS = 15.9       # outer bull (25 ring)
TRIPLE_INNER = 99.0
TRIPLE_OUTER = 107.0
DOUBLE_INNER = 162.0
DOUBLE_OUTER = 170.0

SECTOR_ANGLE = 2 * np.pi / 20  # 18 degrees per sector

# The "top" of the board (sector 20) has its center at -90 degrees (straight up).
# Wire boundaries sit at half-sector offsets from sector centers.
# Sector 20 center is at angle index 0, so the wire between 5 and 20 is at
# -0.5 * SECTOR_ANGLE from sector-20 center, etc.


def sector_center_angle(sector_idx: int) -> float:
    """Angle (radians, CW from top) of sector center. idx 0 = sector 20."""
    return -np.pi / 2 + sector_idx * SECTOR_ANGLE


def wire_angle(sector_idx: int) -> float:
    """Angle of the wire boundary to the LEFT of sector[sector_idx] (CW)."""
    return sector_center_angle(sector_idx) - SECTOR_ANGLE / 2


def point_at(radius: float, angle: float) -> tuple[float, float]:
    """Convert polar (radius, angle) to board XZ coordinates."""
    x = radius * np.cos(angle)
    z = radius * np.sin(angle)
    return (x, z)


# ─── Build the list of labelable points ──────────────────────────────────────

def build_point_list() -> list[dict]:
    """Return a list of {'name': str, 'world': (X, 0, Z)} dicts."""
    points = []

    # Bullseye center
    points.append({"name": "Bullseye center", "world": (0.0, 0.0, 0.0)})

    # Wire-ring intersections at the OUTER DOUBLE ring for each sector boundary
    # These are the 20 radial wire endpoints — very visible on the board edge.
    for i in range(20):
        angle = wire_angle(i)
        x, z = point_at(DOUBLE_OUTER, angle)
        left_num = NUMBERS[i]
        right_num = NUMBERS[(i - 1) % 20]
        name = f"Double outer wire {right_num}/{left_num}"
        points.append({"name": name, "world": (x, 0.0, z)})

    # Wire-ring intersections at the OUTER TRIPLE ring
    for i in range(20):
        angle = wire_angle(i)
        x, z = point_at(TRIPLE_OUTER, angle)
        left_num = NUMBERS[i]
        right_num = NUMBERS[(i - 1) % 20]
        name = f"Triple outer wire {right_num}/{left_num}"
        points.append({"name": name, "world": (x, 0.0, z)})

    # Sector centers at the outer double ring (center of each number's double)
    for i, num in enumerate(NUMBERS):
        angle = sector_center_angle(i)
        x, z = point_at(DOUBLE_OUTER, angle)
        name = f"Double outer center {num}"
        points.append({"name": name, "world": (x, 0.0, z)})

    return points


# ─── Homography calibration (planar target) ──────────────────────────────────

def compute_homography(world_pts: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    """Compute 3x3 homography from board-plane (X, Z) → camera pixel.

    world_pts: (N, 3) with Y=0 for all points.
    image_pts: (N, 2)
    Returns H (3, 3) such that pixel ~ H @ [X, Z, 1]^T.
    """
    # Use only (X, Z) since all points are coplanar at Y=0
    src = np.column_stack([world_pts[:, 0], world_pts[:, 2]])  # (N, 2)
    dst = image_pts  # (N, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    # Reprojection error
    n = src.shape[0]
    ones = np.ones((n, 1))
    src_h = np.hstack([src, ones])
    proj = (H @ src_h.T).T
    proj_2d = proj[:, :2] / proj[:, 2:3]
    errors = np.linalg.norm(proj_2d - dst, axis=1)
    inliers = mask.ravel().sum() if mask is not None else n
    print(f"  Reprojection error: mean={errors.mean():.2f}px, max={errors.max():.2f}px, inliers={inliers}/{n}")

    return H


# ─── Interactive labeler ─────────────────────────────────────────────────────

class PointLabeler:
    def __init__(self, image: np.ndarray, point_list: list[dict], cam_name: str):
        self.orig = image.copy()
        self.display = image.copy()
        self.points = point_list
        self.cam_name = cam_name
        self.current_idx = 0
        self.labeled: list[dict] = []  # {'idx': int, 'pixel': (u,v), 'world': (X,Y,Z)}
        self.done = False
        self.aborted = False

    def _redraw(self):
        self.display = self.orig.copy()
        h, w = self.display.shape[:2]

        # Draw labeled points
        for item in self.labeled:
            px = item["pixel"]
            cv2.circle(self.display, px, 5, (0, 255, 0), -1)
            cv2.circle(self.display, px, 7, (0, 0, 0), 1)
            cv2.putText(self.display, str(item["idx"] + 1), (px[0] + 8, px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Status bar
        cv2.rectangle(self.display, (0, 0), (w, 35), (0, 0, 0), -1)
        if self.current_idx < len(self.points):
            name = self.points[self.current_idx]["name"]
            status = f"[{self.cam_name}] Click: {name} ({self.current_idx+1}/{len(self.points)}) | {len(self.labeled)} labeled | Right-click=skip | u=undo | Enter=done"
        else:
            status = f"[{self.cam_name}] All points shown. {len(self.labeled)} labeled. Press Enter to finish."
        cv2.putText(self.display, status, (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _on_mouse(self, event, x, y, flags, param):
        if self.current_idx >= len(self.points):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            pt = self.points[self.current_idx]
            self.labeled.append({
                "idx": self.current_idx,
                "pixel": (x, y),
                "world": pt["world"],
                "name": pt["name"],
            })
            print(f"    [{self.current_idx+1}] {pt['name']} -> ({x}, {y})")
            self.current_idx += 1
            self._redraw()
            cv2.imshow(self.cam_name, self.display)

        elif event == cv2.EVENT_RBUTTONDOWN:
            pt = self.points[self.current_idx]
            print(f"    [{self.current_idx+1}] {pt['name']} -> SKIPPED")
            self.current_idx += 1
            self._redraw()
            cv2.imshow(self.cam_name, self.display)

    def run(self) -> list[dict]:
        win = self.cam_name
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._on_mouse)
        self._redraw()
        cv2.imshow(win, self.display)

        print(f"\n  Labeling {win} — left-click to mark, right-click to skip, 'u' to undo, Enter when done.\n")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                self.aborted = True
                break
            elif key == 13:  # Enter
                if len(self.labeled) >= 4:
                    break
                else:
                    print(f"    Need at least 4 labeled points, have {len(self.labeled)}.")
            elif key == ord('u'):
                if self.labeled:
                    removed = self.labeled.pop()
                    self.current_idx = removed["idx"]
                    print(f"    Undid: {removed['name']}")
                    self._redraw()
                    cv2.imshow(win, self.display)

        cv2.destroyWindow(win)
        return self.labeled


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent
    screenshot_dir = script_dir / "screenshots"

    cameras = ["top_left", "top_right", "bottom_right"]

    # Pick one image per camera (first timestamp)
    cam_images = {}
    for cam in cameras:
        pattern = f"*_{cam}.png"
        files = sorted(screenshot_dir.glob(pattern))
        if not files:
            print(f"No images found for {cam}")
            sys.exit(1)
        cam_images[cam] = files[0]
        print(f"{cam}: {files[0].name} ({len(files)} images total)")

    point_list = build_point_list()
    print(f"\n{len(point_list)} labelable points defined (bullseye + wire intersections + sector centers).")
    print("You only need to label >= 4 per camera. More = better accuracy.")
    print("Tip: bullseye + the 20 outer double wire intersections are easiest to spot.\n")

    results = {}
    labels_for_save = {}

    for cam in cameras:
        img = cv2.imread(str(cam_images[cam]))
        if img is None:
            print(f"Failed to load {cam_images[cam]}")
            sys.exit(1)

        labeler = PointLabeler(img, point_list, cam)
        labeled = labeler.run()

        if labeler.aborted:
            print("Aborted.")
            sys.exit(0)

        print(f"\n  {cam}: {len(labeled)} points labeled.")

        # Build correspondences
        world_pts = np.array([l["world"] for l in labeled], dtype=np.float64)
        image_pts = np.array([l["pixel"] for l in labeled], dtype=np.float64)

        # Homography (planar target, all Y=0)
        print(f"  Computing homography for {cam}...")
        H = compute_homography(world_pts, image_pts)

        print(f"\n  H (board XZ -> pixel):\n{H}")
        print()

        results[cam] = {"H": H}
        labels_for_save[cam] = {
            "image": cam_images[cam].name,
            "points": [
                {"name": l["name"], "pixel": list(l["pixel"]), "world": list(l["world"])}
                for l in labeled
            ],
        }

    # ─── Save ─────────────────────────────────────────────────────────────────
    out_npz = script_dir / "camera_data_3cam.npz"
    save_dict = {}
    for cam in cameras:
        save_dict[f"H_{cam}"] = results[cam]["H"]
    np.savez(str(out_npz), **save_dict)
    print(f"Homographies saved to: {out_npz}")

    # Save labels as JSON for reproducibility
    out_json = script_dir / "calibration_labels.json"
    with open(out_json, "w") as f:
        json.dump(labels_for_save, f, indent=2)
    print(f"Labels saved to: {out_json}")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cam in cameras:
        H = results[cam]["H"]
        print(f"\n--- {cam} ---")
        print(f"  H (board XZ -> pixel):\n{H}")


if __name__ == "__main__":
    main()
