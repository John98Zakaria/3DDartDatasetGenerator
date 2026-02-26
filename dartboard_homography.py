"""
Dartboard Homography Viewer — Stereo Edition

Uses known camera projection matrices (camera_data.npz) and both camera
perspectives for stereo triangulation of the dart landing position.

Three windows: Camera Left, Camera Right, Dartboard

Phase 1 — Board calibration:
    Click a recognizable point on the LEFT camera image — the projection
    matrix auto-computes the board-plane (X,Z) world coordinates.
    Then click the same point on the dartboard.jpg image.
    After 3+ pairs, press 'h' to fit the board-plane → dartboard.jpg mapping.
    A validation grid is projected; press 'a' to accept, 'r' to redo.

Phase 2 — Stereo dart mapping:
    Click the dart tip on the LEFT camera — an epipolar line is drawn on
    the RIGHT camera to guide your second click.
    Click the dart tip on the RIGHT camera — the 3D position is
    triangulated, projected onto the board plane, and mapped to dartboard.jpg.

Keys:
    h — compute mapping (calibration phase, needs 3+ pairs)
    a — accept validation
    r — redo / return to calibration
    u — undo last point
    c — clear dart markers
    q — quit
"""

import os
import sys

import cv2
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "camera_data.npz")
IMG_LEFT = os.path.join(SCRIPT_DIR, "render_Camera_Left.png")
IMG_RIGHT = os.path.join(SCRIPT_DIR, "render_Camera_Right.png")
BOARD_IMG_PATH = os.path.join(SCRIPT_DIR, "assets", "dartboard.jpg")

# ─── Window names ─────────────────────────────────────────────────────────────
WIN_LEFT = "Camera Left"
WIN_RIGHT = "Camera Right"
WIN_BOARD = "Dartboard"

# ─── Board plane: Y = 0 in world coordinates ─────────────────────────────────
BOARD_Y = 0.0

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def board_plane_homography(P, board_y=BOARD_Y):
    """3×3 homography mapping board-plane [X, Z, 1] → camera pixel.

    For a point at (X, board_y, Z):
        pixel ~ P @ [X, board_y, Z, 1]^T
              = P[:,0]*X + P[:,2]*Z + (P[:,1]*board_y + P[:,3])
    So H = [P[:,0] | P[:,2] | P[:,1]*board_y + P[:,3]]
    """
    return np.column_stack([P[:, 0], P[:, 2], P[:, 1] * board_y + P[:, 3]])


def camera_pixel_to_board_xz(pixel, H):
    """Invert the board-plane homography: camera pixel → (X, Z)."""
    H_inv = np.linalg.inv(H)
    p = np.array([pixel[0], pixel[1], 1.0])
    xz = H_inv @ p
    xz /= xz[2]
    return xz[0], xz[1]


def triangulate_point(P1, P2, pt1, pt2):
    """DLT triangulation from two 2D observations → 3D world point."""
    A = np.array([
        pt1[0] * P1[2, :] - P1[0, :],
        pt1[1] * P1[2, :] - P1[1, :],
        pt2[0] * P2[2, :] - P2[0, :],
        pt2[1] * P2[2, :] - P2[1, :],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]
    return X[:3]


def fundamental_from_projections(P1, P2):
    """Fundamental matrix F such that x2^T F x1 = 0."""
    _, _, Vt = np.linalg.svd(P1)
    C = Vt[-1]
    e2 = P2 @ C
    e2 /= e2[2]
    e2x = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0],
    ])
    F = e2x @ P2 @ np.linalg.pinv(P1)
    F /= np.linalg.norm(F)
    return F


def epipolar_line_endpoints(line, w, h):
    """Clip an epipolar line ax+by+c=0 to the image rectangle."""
    a, b, c = line
    pts = []
    if abs(b) > 1e-12:
        y = -c / b
        if 0 <= y < h:
            pts.append((0.0, y))
        y = -(a * (w - 1) + c) / b
        if 0 <= y < h:
            pts.append((float(w - 1), y))
    if abs(a) > 1e-12:
        x = -c / a
        if 0 <= x < w:
            pts.append((x, 0.0))
        x = -(b * (h - 1) + c) / a
        if 0 <= x < w:
            pts.append((x, float(h - 1)))
    if len(pts) < 2:
        return None
    pts = list(set(pts))
    if len(pts) < 2:
        return None
    best = max(
        ((pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )
    return (
        (int(round(best[0][0])), int(round(best[0][1]))),
        (int(round(best[1][0])), int(round(best[1][1]))),
    )


def xz_to_board_pixel(X, Z, T):
    """Apply affine transform T (2×3) to map board-plane (X,Z) → dartboard pixel."""
    src = np.array([[[X, Z]]], dtype=np.float64)
    dst = cv2.transform(src, T)
    return int(round(dst[0, 0, 0])), int(round(dst[0, 0, 1]))


# ─── Main viewer class ───────────────────────────────────────────────────────

class DartboardViewer:
    def __init__(self):
        # Load camera data
        if not os.path.isfile(NPZ_PATH):
            sys.exit(f"Error: {NPZ_PATH} not found. Run get_camera_matrices.py in Blender first.")

        data = np.load(NPZ_PATH, allow_pickle=True)
        self.P1, self.P2 = data["P1"], data["P2"]

        # Board-plane homographies: [X, Z, 1] → camera pixel
        self.H_left = board_plane_homography(self.P1)
        self.H_right = board_plane_homography(self.P2)

        # Fundamental matrix for epipolar guidance
        self.F = fundamental_from_projections(self.P1, self.P2)

        # Load images
        self.left_orig = cv2.imread(IMG_LEFT)
        self.right_orig = cv2.imread(IMG_RIGHT)
        self.board_orig = cv2.imread(BOARD_IMG_PATH)
        if self.left_orig is None:
            sys.exit(f"Cannot load {IMG_LEFT}")
        if self.right_orig is None:
            sys.exit(f"Cannot load {IMG_RIGHT}")
        if self.board_orig is None:
            sys.exit(f"Cannot load {BOARD_IMG_PATH}")

        # Working copies
        self.left_img = self.left_orig.copy()
        self.right_img = self.right_orig.copy()
        self.board_img = self.board_orig.copy()

        # ── Calibration state ──
        # Pairs of board-plane (X,Z) computed from camera clicks
        # and corresponding (u,v) dartboard.jpg clicks
        self.calib_xz: list[tuple[float, float]] = []
        self.calib_uv: list[tuple[int, int]] = []
        self.calib_cam_px: list[tuple[int, int]] = []  # for redrawing markers

        self.expecting = "camera"  # "camera" or "board"
        self.pending_xz: tuple[float, float] | None = None
        self.pending_cam_px: tuple[int, int] | None = None

        # Affine transform: board-plane (X,Z) → dartboard.jpg (u,v)  [2×3]
        self.T: np.ndarray | None = None

        # ── Phase ──
        self.phase = "calibration"  # "calibration" | "validate" | "mapping"

        # ── Stereo dart state ──
        self.dart_left_click: tuple[int, int] | None = None
        self.dart_board_pts: list[tuple[int, int]] = []  # mapped positions on dartboard

    # ------------------------------------------------------------------ #
    #  Drawing
    # ------------------------------------------------------------------ #

    def _redraw(self):
        self.left_img = self.left_orig.copy()
        self.right_img = self.right_orig.copy()
        self.board_img = self.board_orig.copy()

        if self.phase == "calibration":
            self._draw_calib_markers()
            self._draw_calib_status()
        elif self.phase == "validate":
            self._draw_validation()
        elif self.phase == "mapping":
            self._draw_mapping()

        cv2.imshow(WIN_LEFT, self.left_img)
        cv2.imshow(WIN_RIGHT, self.right_img)
        cv2.imshow(WIN_BOARD, self.board_img)

    def _draw_calib_markers(self):
        # Completed pairs
        for i, (cam_px, uv) in enumerate(zip(self.calib_cam_px, self.calib_uv)):
            label = str(i + 1)
            cv2.circle(self.left_img, cam_px, 6, GREEN, -1)
            cv2.circle(self.left_img, cam_px, 8, (0, 0, 0), 1)
            cv2.putText(self.left_img, label, (cam_px[0] + 10, cam_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            cv2.circle(self.board_img, uv, 6, GREEN, -1)
            cv2.circle(self.board_img, uv, 8, (0, 0, 0), 1)
            cv2.putText(self.board_img, label, (uv[0] + 10, uv[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

            # Also project to right camera for reference
            xz = self.calib_xz[i]
            rpt_h = self.H_right @ np.array([xz[0], xz[1], 1.0])
            rpt = (int(round(rpt_h[0] / rpt_h[2])), int(round(rpt_h[1] / rpt_h[2])))
            h_r, w_r = self.right_img.shape[:2]
            if 0 <= rpt[0] < w_r and 0 <= rpt[1] < h_r:
                cv2.circle(self.right_img, rpt, 6, GREEN, -1)
                cv2.putText(self.right_img, label, (rpt[0] + 10, rpt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        # Pending unpaired camera click
        if self.pending_cam_px is not None:
            label = str(len(self.calib_cam_px) + 1)
            cv2.circle(self.left_img, self.pending_cam_px, 6, YELLOW, -1)
            cv2.putText(self.left_img, label, (self.pending_cam_px[0] + 10,
                                                self.pending_cam_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    def _draw_calib_status(self):
        n = len(self.calib_xz)
        if self.expecting == "camera":
            status = f"Click LEFT CAMERA point #{n + 1}  |  {n} pairs  |  'h'=compute  'u'=undo  'q'=quit"
        else:
            status = f"Click DARTBOARD point #{n + 1}  |  {n} pairs  |  'u'=undo  'q'=quit"

        for img in (self.left_img, self.right_img):
            cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(self.left_img, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(self.right_img, "Right camera (reference only during calibration)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    def _draw_validation(self):
        """Project a grid of board-plane points onto all three views."""
        # Determine range from calibration points
        xs = [xz[0] for xz in self.calib_xz]
        zs = [xz[1] for xz in self.calib_xz]
        margin = max(max(xs) - min(xs), max(zs) - min(zs)) * 0.3
        x_lo, x_hi = min(xs) - margin, max(xs) + margin
        z_lo, z_hi = min(zs) - margin, max(zs) + margin

        h_l, w_l = self.left_img.shape[:2]
        h_r, w_r = self.right_img.shape[:2]
        h_b, w_b = self.board_img.shape[:2]

        for X in np.linspace(x_lo, x_hi, 12):
            for Z in np.linspace(z_lo, z_hi, 12):
                board_pt = np.array([X, Z, 1.0])

                # Left camera
                lp_h = self.H_left @ board_pt
                lp = (int(lp_h[0] / lp_h[2]), int(lp_h[1] / lp_h[2]))
                if 0 <= lp[0] < w_l and 0 <= lp[1] < h_l:
                    cv2.circle(self.left_img, lp, 3, MAGENTA, -1)

                # Right camera
                rp_h = self.H_right @ board_pt
                rp = (int(rp_h[0] / rp_h[2]), int(rp_h[1] / rp_h[2]))
                if 0 <= rp[0] < w_r and 0 <= rp[1] < h_r:
                    cv2.circle(self.right_img, rp, 3, MAGENTA, -1)

                # Dartboard.jpg
                bu, bv = xz_to_board_pixel(X, Z, self.T)
                if 0 <= bu < w_b and 0 <= bv < h_b:
                    cv2.circle(self.board_img, (bu, bv), 3, MAGENTA, -1)

        # Redraw calibration point markers on top
        self._draw_calib_markers()

        # Status
        for img in (self.left_img, self.right_img):
            cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(self.left_img, "Validation grid shown. 'a'=accept  'r'=redo", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(self.right_img, "Validation grid (right camera)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    def _draw_mapping(self):
        """Draw dart markers and epipolar guidance for stereo mode."""
        # Existing mapped points on dartboard
        for bpt in self.dart_board_pts:
            cv2.drawMarker(self.board_img, bpt, RED, cv2.MARKER_CROSS, 30, 2)
            cv2.circle(self.board_img, bpt, 12, MAGENTA, 2)

        # If left click pending, show marker + epipolar line on right
        if self.dart_left_click is not None:
            pt = self.dart_left_click
            cv2.drawMarker(self.left_img, pt, RED, cv2.MARKER_CROSS, 20, 2)
            cv2.circle(self.left_img, pt, 8, YELLOW, 2)

            # Epipolar line on right image
            p = np.array([pt[0], pt[1], 1.0])
            l = self.F @ p
            h_r, w_r = self.right_img.shape[:2]
            ep = epipolar_line_endpoints(l, w_r, h_r)
            if ep:
                cv2.line(self.right_img, ep[0], ep[1], GREEN, 2)

            # Also show single-camera estimate on dartboard (as a hint)
            xz = camera_pixel_to_board_xz(pt, self.H_left)
            bu, bv = xz_to_board_pixel(xz[0], xz[1], self.T)
            h_b, w_b = self.board_img.shape[:2]
            if 0 <= bu < w_b and 0 <= bv < h_b:
                cv2.circle(self.board_img, (bu, bv), 8, CYAN, 1)  # thin cyan = estimate

        # Status bar
        if self.dart_left_click is None:
            status = "Click dart on LEFT camera  |  'c'=clear  'r'=recalibrate  'q'=quit"
        else:
            status = "Click dart on RIGHT camera (follow epipolar line)  |  'c'=clear  'q'=quit"

        for img in (self.left_img, self.right_img):
            cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(self.left_img, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        right_status = "Click here after left (epipolar line = green)" if self.dart_left_click else "Stereo dart mapping"
        cv2.putText(self.right_img, right_status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Mouse callbacks
    # ------------------------------------------------------------------ #

    def _on_left_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.phase == "calibration" and self.expecting == "camera":
            xz = camera_pixel_to_board_xz((x, y), self.H_left)
            self.pending_xz = xz
            self.pending_cam_px = (x, y)
            self.expecting = "board"
            print(f"  Left camera ({x}, {y}) -> board plane (X={xz[0]:.2f}, Z={xz[1]:.2f})")
            print(f"  Now click the same point on the dartboard image.")
            self._redraw()

        elif self.phase == "mapping":
            if self.dart_left_click is None:
                self.dart_left_click = (x, y)
                xz = camera_pixel_to_board_xz((x, y), self.H_left)
                print(f"  Left click ({x}, {y}) -> board XZ ({xz[0]:.2f}, {xz[1]:.2f})")
                print(f"  Now click the same point on the RIGHT camera.")
                self._redraw()

    def _on_right_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.phase == "mapping" and self.dart_left_click is not None:
            # Stereo triangulation
            pt3d = triangulate_point(self.P1, self.P2, self.dart_left_click, (x, y))
            board_x, board_z = pt3d[0], pt3d[2]
            bu, bv = xz_to_board_pixel(board_x, board_z, self.T)

            print(f"  Right click ({x}, {y})")
            print(f"  Triangulated 3D: ({pt3d[0]:.3f}, {pt3d[1]:.3f}, {pt3d[2]:.3f})")
            print(f"  Board plane: X={board_x:.3f}, Z={board_z:.3f}  (Y offset from plane: {pt3d[1] - BOARD_Y:.3f})")
            print(f"  Dartboard pixel: ({bu}, {bv})")

            self.dart_board_pts.append((bu, bv))
            self.dart_left_click = None
            self._redraw()

    def _on_board_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.phase == "calibration" and self.expecting == "board":
            self.calib_xz.append(self.pending_xz)
            self.calib_uv.append((x, y))
            self.calib_cam_px.append(self.pending_cam_px)
            self.pending_xz = None
            self.pending_cam_px = None
            self.expecting = "camera"
            n = len(self.calib_xz)
            print(f"  Dartboard click ({x}, {y}) — pair #{n} complete.")
            self._redraw()

    # ------------------------------------------------------------------ #
    #  Affine mapping computation
    # ------------------------------------------------------------------ #

    def _compute_affine(self) -> bool:
        n = len(self.calib_xz)
        if n < 3:
            print(f"Need at least 3 pairs, have {n}.")
            return False

        src = np.array(self.calib_xz, dtype=np.float64)
        dst = np.array(self.calib_uv, dtype=np.float64)

        if n == 3:
            self.T = cv2.getAffineTransform(
                src.astype(np.float32), dst.astype(np.float32)
            )
        else:
            # Least-squares affine from n > 3 pairs
            # [u, v]^T = T @ [X, Z, 1]^T
            A = np.zeros((2 * n, 6))
            b = np.zeros(2 * n)
            for i in range(n):
                X, Z = src[i]
                u, v = dst[i]
                A[2 * i] = [X, Z, 1, 0, 0, 0]
                A[2 * i + 1] = [0, 0, 0, X, Z, 1]
                b[2 * i] = u
                b[2 * i + 1] = v
            params, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
            self.T = params.reshape(2, 3)
            if len(residuals) > 0:
                print(f"  Affine fit residual: {residuals[0]:.4f}")

        print(f"Affine transform (board XZ -> dartboard pixel) from {n} pairs:")
        print(f"  {self.T}")
        return True

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self):
        cv2.namedWindow(WIN_LEFT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_RIGHT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_BOARD, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN_LEFT, self._on_left_click)
        cv2.setMouseCallback(WIN_RIGHT, self._on_right_click)
        cv2.setMouseCallback(WIN_BOARD, self._on_board_click)

        self._redraw()
        print("=== Phase 1: Board Calibration ===")
        print("Camera matrices loaded — clicks on the left camera are auto-mapped")
        print("to board-plane (X,Z) world coordinates via the projection matrix.")
        print()
        print("  1. Click a recognizable point on the LEFT camera image")
        print("  2. Click the same point on the DARTBOARD image")
        print("  3. Repeat for 3+ pairs (bullseye, wire junctions, number positions)")
        print("  4. Press 'h' to compute the mapping")
        print()

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                break

            # ── Calibration ── #
            if self.phase == "calibration":
                if key == ord('h'):
                    if self._compute_affine():
                        self.phase = "validate"
                        self._redraw()
                        print("\nValidation: magenta grid projected onto all views.")
                        print("Press 'a' to accept, 'r' to redo.\n")
                elif key == ord('u'):
                    if self.expecting == "board" and self.pending_xz is not None:
                        self.pending_xz = None
                        self.pending_cam_px = None
                        self.expecting = "camera"
                    elif self.calib_xz:
                        self.calib_xz.pop()
                        self.calib_uv.pop()
                        self.calib_cam_px.pop()
                    self._redraw()

            # ── Validation ── #
            elif self.phase == "validate":
                if key == ord('a'):
                    self.phase = "mapping"
                    self.dart_board_pts = []
                    self.dart_left_click = None
                    self._redraw()
                    print("=== Phase 2: Stereo Dart Mapping ===")
                    print("  1. Click the dart tip on the LEFT camera")
                    print("  2. Click the dart tip on the RIGHT camera (follow green epipolar line)")
                    print("  3. Triangulated position is shown on the dartboard")
                    print()
                    print("Keys: 'c'=clear  'r'=recalibrate  'q'=quit")
                    print()
                elif key == ord('r'):
                    self.phase = "calibration"
                    self.T = None
                    self._redraw()
                    print("Returning to calibration.\n")

            # ── Mapping ── #
            elif self.phase == "mapping":
                if key == ord('c'):
                    self.dart_board_pts = []
                    self.dart_left_click = None
                    self._redraw()
                    print("Cleared.\n")
                elif key == ord('r'):
                    self.phase = "calibration"
                    self.T = None
                    self.expecting = "camera"
                    self.dart_board_pts = []
                    self.dart_left_click = None
                    self._redraw()
                    print("\n=== Back to Calibration ===\n")

        cv2.destroyAllWindows()


def main():
    viewer = DartboardViewer()
    viewer.run()


if __name__ == "__main__":
    main()
