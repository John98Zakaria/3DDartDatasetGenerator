"""
Epipolar Line Visualizer
========================
Click a point on one camera view to draw the corresponding epipolar line
on the other view.

Usage:  python epipolar_viewer.py
Keys:   c = clear annotations,  q = quit
"""

import sys
import os
import numpy as np
import cv2

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "camera_data.npz")
IMG_LEFT = os.path.join(SCRIPT_DIR, "render_Camera_Left.png")
IMG_RIGHT = os.path.join(SCRIPT_DIR, "render_Camera_Right.png")

WIN_LEFT = "Camera Left"
WIN_RIGHT = "Camera Right"

LINE_COLOR = (0, 255, 0)      # green epipolar line
POINT_COLOR = (0, 0, 255)     # red clicked point
LINE_THICKNESS = 2
POINT_RADIUS = 5


# ─── Math helpers ─────────────────────────────────────────────────────────────

def fundamental_from_projections(P1, P2):
    """Compute the fundamental matrix F such that x2^T F x1 = 0.

    F = [e2]_x  @  P2  @  pinv(P1)
    where e2 = P2 @ null(P1) is the epipole in image 2.
    """
    # Null space of P1 (camera center in homogeneous coords)
    _, _, Vt = np.linalg.svd(P1)
    C = Vt[-1]  # last row of Vt = right null vector

    # Epipole in image 2
    e2 = P2 @ C
    e2 /= e2[2]  # normalise

    # Skew-symmetric matrix of e2
    e2x = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0],
    ])

    F = e2x @ P2 @ np.linalg.pinv(P1)
    # Normalise so that ||F||_fro = 1
    F /= np.linalg.norm(F)
    return F


def epipolar_line_endpoints(line, w, h):
    """Given a line l = (a, b, c) with ax + by + c = 0, return two endpoints
    clipped to the image rectangle [0, w) x [0, h)."""
    a, b, c = line
    pts = []
    if abs(b) > 1e-12:
        # intersection with x = 0
        y = -c / b
        if 0 <= y < h:
            pts.append((0, y))
        # intersection with x = w-1
        y = -(a * (w - 1) + c) / b
        if 0 <= y < h:
            pts.append(((w - 1), y))
    if abs(a) > 1e-12:
        # intersection with y = 0
        x = -c / a
        if 0 <= x < w:
            pts.append((x, 0))
        # intersection with y = h-1
        x = -(b * (h - 1) + c) / a
        if 0 <= x < w:
            pts.append((x, (h - 1)))
    if len(pts) < 2:
        return None
    # Deduplicate and pick the two furthest apart
    pts = list(set(pts))
    if len(pts) < 2:
        return None
    best = max(
        ((pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )
    return tuple(int(round(v)) for v in best[0]), tuple(int(round(v)) for v in best[1])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load data
    if not os.path.isfile(NPZ_PATH):
        print(f"Error: {NPZ_PATH} not found. Run get_camera_matrices.py in Blender first.")
        sys.exit(1)

    data = np.load(NPZ_PATH, allow_pickle=True)
    P1, P2 = data["P1"], data["P2"]
    cam1_name = str(data["cam1_name"])
    cam2_name = str(data["cam2_name"])

    img_left_orig = cv2.imread(IMG_LEFT)
    img_right_orig = cv2.imread(IMG_RIGHT)
    if img_left_orig is None or img_right_orig is None:
        print("Error: could not load one or both images.")
        sys.exit(1)

    # Compute fundamental matrix
    F = fundamental_from_projections(P1, P2)
    print(f"Fundamental matrix F (from {cam1_name} -> {cam2_name}):")
    print(F)
    print()

    # Working copies for drawing
    img_left = img_left_orig.copy()
    img_right = img_right_orig.copy()

    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    # State for two-click line drawing: stores first click as (x, y) or None
    left_first_click = [None]   # mutable container for nested closure access
    right_first_click = [None]

    POINT2_COLOR = (255, 0, 0)      # blue for second point
    DRAWN_LINE_COLOR = (0, 255, 255) # yellow for the drawn line on source image
    INTERSECT_COLOR = (0, 165, 255)  # orange for intersection point

    def line_intersection(l1, l2):
        """Intersection of two 2D lines in homogeneous form: p = l1 x l2."""
        p = np.cross(l1, l2)
        if abs(p[2]) < 1e-12:
            return None  # parallel
        return p[:2] / p[2]

    def on_click_left(event, x, y, flags, param):
        nonlocal img_left, img_right
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if left_first_click[0] is None:
            # First click: just mark the point, wait for second
            left_first_click[0] = (x, y)
            img_left = img_left_orig.copy()
            img_right = img_right_orig.copy()
            cv2.circle(img_left, (x, y), POINT_RADIUS, POINT_COLOR, -1)
            cv2.imshow(WIN_LEFT, img_left)
            cv2.imshow(WIN_RIGHT, img_right)
            print(f"Left 1st click ({x}, {y}) — click again to define line")
        else:
            # Second click: draw line on left, epipolar lines on right
            x0, y0 = left_first_click[0]
            left_first_click[0] = None

            img_left = img_left_orig.copy()
            img_right = img_right_orig.copy()

            # Draw the two points and line on left image
            cv2.circle(img_left, (x0, y0), POINT_RADIUS, POINT_COLOR, -1)
            cv2.circle(img_left, (x, y), POINT_RADIUS, POINT2_COLOR, -1)
            cv2.line(img_left, (x0, y0), (x, y), DRAWN_LINE_COLOR, LINE_THICKNESS)

            # Epipolar lines in right image for both endpoints
            p1 = np.array([x0, y0, 1.0])
            p2 = np.array([x, y, 1.0])
            l1 = F @ p1
            l2 = F @ p2
            ep1 = epipolar_line_endpoints(l1, w_r, h_r)
            ep2 = epipolar_line_endpoints(l2, w_r, h_r)
            if ep1:
                cv2.line(img_right, ep1[0], ep1[1], LINE_COLOR, LINE_THICKNESS)
            if ep2:
                cv2.line(img_right, ep2[0], ep2[1], LINE_COLOR, LINE_THICKNESS)

            # Mark intersection of the two epipolar lines
            ix = line_intersection(l1, l2)
            if ix is not None and 0 <= ix[0] < w_r and 0 <= ix[1] < h_r:
                cv2.circle(img_right, (int(round(ix[0])), int(round(ix[1]))),
                           POINT_RADIUS + 2, INTERSECT_COLOR, -1)
                print(f"  Epipolar intersection in right image: ({ix[0]:.1f}, {ix[1]:.1f})")

            cv2.imshow(WIN_LEFT, img_left)
            cv2.imshow(WIN_RIGHT, img_right)
            print(f"Left line ({x0},{y0})->({x},{y})")

    def on_click_right(event, x, y, flags, param):
        nonlocal img_left, img_right
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if right_first_click[0] is None:
            # First click: just mark the point, wait for second
            right_first_click[0] = (x, y)
            img_left = img_left_orig.copy()
            img_right = img_right_orig.copy()
            cv2.circle(img_right, (x, y), POINT_RADIUS, POINT_COLOR, -1)
            cv2.imshow(WIN_LEFT, img_left)
            cv2.imshow(WIN_RIGHT, img_right)
            print(f"Right 1st click ({x}, {y}) — click again to define line")
        else:
            # Second click: draw line on right, epipolar lines on left
            x0, y0 = right_first_click[0]
            right_first_click[0] = None

            img_left = img_left_orig.copy()
            img_right = img_right_orig.copy()

            # Draw the two points and line on right image
            cv2.circle(img_right, (x0, y0), POINT_RADIUS, POINT_COLOR, -1)
            cv2.circle(img_right, (x, y), POINT_RADIUS, POINT2_COLOR, -1)
            cv2.line(img_right, (x0, y0), (x, y), DRAWN_LINE_COLOR, LINE_THICKNESS)

            # Epipolar lines in left image for both endpoints
            p1 = np.array([x0, y0, 1.0])
            p2 = np.array([x, y, 1.0])
            l1 = F.T @ p1
            l2 = F.T @ p2
            ep1 = epipolar_line_endpoints(l1, w_l, h_l)
            ep2 = epipolar_line_endpoints(l2, w_l, h_l)
            if ep1:
                cv2.line(img_left, ep1[0], ep1[1], LINE_COLOR, LINE_THICKNESS)
            if ep2:
                cv2.line(img_left, ep2[0], ep2[1], LINE_COLOR, LINE_THICKNESS)

            # Mark intersection of the two epipolar lines
            ix = line_intersection(l1, l2)
            if ix is not None and 0 <= ix[0] < w_l and 0 <= ix[1] < h_l:
                cv2.circle(img_left, (int(round(ix[0])), int(round(ix[1]))),
                           POINT_RADIUS + 2, INTERSECT_COLOR, -1)
                print(f"  Epipolar intersection in left image: ({ix[0]:.1f}, {ix[1]:.1f})")

            cv2.imshow(WIN_LEFT, img_left)
            cv2.imshow(WIN_RIGHT, img_right)
            print(f"Right line ({x0},{y0})->({x},{y})")

    # Create windows
    cv2.namedWindow(WIN_LEFT, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RIGHT, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_LEFT, on_click_left)
    cv2.setMouseCallback(WIN_RIGHT, on_click_right)

    cv2.imshow(WIN_LEFT, img_left)
    cv2.imshow(WIN_RIGHT, img_right)

    print("Epipolar Viewer ready.")
    print("  Two clicks on one image define a line.")
    print("  Epipolar lines + their intersection are drawn on the other image.")
    print("  Press 'c' to clear, 'q' to quit.")
    print()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            left_first_click[0] = None
            right_first_click[0] = None
            img_left = img_left_orig.copy()
            img_right = img_right_orig.copy()
            cv2.imshow(WIN_LEFT, img_left)
            cv2.imshow(WIN_RIGHT, img_right)
            print("Cleared.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
