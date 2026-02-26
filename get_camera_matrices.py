import bpy
import mathutils
import numpy as np
import os

def get_intrinsic_matrix(camera_data, scene):
    """Build the 3x3 intrinsic (calibration) matrix K from Blender camera parameters."""
    render = scene.render
    # Resolution in pixels
    res_x = render.resolution_x * render.resolution_percentage / 100
    res_y = render.resolution_y * render.resolution_percentage / 100

    # Focal length in mm and sensor size in mm
    focal_mm = camera_data.lens
    sensor_width = camera_data.sensor_width
    sensor_height = camera_data.sensor_height

    # Pixel size depends on sensor fit mode
    if camera_data.sensor_fit == 'VERTICAL':
        # Sensor height maps to res_y
        pixel_aspect = res_y / sensor_height
    else:
        # AUTO or HORIZONTAL: sensor width maps to res_x
        pixel_aspect = res_x / sensor_width

    # Focal length in pixels
    f_x = focal_mm * pixel_aspect
    f_y = focal_mm * pixel_aspect

    # Principal point (center of image)
    cx = res_x / 2.0
    cy = res_y / 2.0

    # Account for shift (Blender uses fraction-of-sensor-size units)
    if camera_data.sensor_fit == 'VERTICAL':
        cx += camera_data.shift_x * res_y
        cy += camera_data.shift_y * res_y
    else:
        cx += camera_data.shift_x * res_x
        cy += camera_data.shift_y * res_x

    K = np.array([
        [f_x,  0,   cx],
        [ 0,  f_y,  cy],
        [ 0,   0,    1],
    ])
    return K


def get_extrinsic_matrix(camera_obj):
    """Get the 4x4 world-to-camera (extrinsic) matrix.

    Blender's matrix_world is object-to-world.
    The extrinsic matrix is the inverse: world-to-camera.
    Convention: OpenCV-style (X-right, Y-down, Z-forward).
    """
    # Blender camera looks down -Z with Y up.
    # To convert to OpenCV convention (Z forward, Y down), flip Y and Z.
    flip = mathutils.Matrix((
        (1,  0,  0, 0),
        (0, -1,  0, 0),
        (0,  0, -1, 0),
        (0,  0,  0, 1),
    ))
    extrinsic = flip @ camera_obj.matrix_world.normalized().inverted()
    return np.array(extrinsic)


def get_projection_matrix(K, extrinsic):
    """Compute the full 3x4 projection matrix P = K @ [R|t]."""
    Rt = extrinsic[:3, :]  # top 3 rows of the 4x4
    P = K @ Rt
    return P


# ─── Main ────────────────────────────────────────────────────────────────────
scene = bpy.context.scene
output_dir = r"C:\Users\John\PycharmProjects\PythonProject"
output_path = os.path.join(output_dir, "camera_matrices.txt")

cameras = sorted(
    [obj for obj in bpy.data.objects if obj.type == 'CAMERA'],
    key=lambda o: o.name,
)

# Collect matrices for .npz export (keyed by camera name order)
npz_data = {}

with open(output_path, "w") as f:
    f.write(f"Found {len(cameras)} camera(s) in the scene.\n\n")

    for cam_obj in cameras:
        cam_data = cam_obj.data
        f.write(f"{'='*60}\n")
        f.write(f"Camera: {cam_obj.name}\n")
        f.write(f"  Focal length : {cam_data.lens:.2f} mm\n")
        f.write(f"  Sensor       : {cam_data.sensor_width:.2f} x {cam_data.sensor_height:.2f} mm ({cam_data.sensor_fit})\n")
        f.write(f"  Shift        : ({cam_data.shift_x:.4f}, {cam_data.shift_y:.4f})\n")
        f.write(f"  Location     : {cam_obj.location}\n")
        f.write(f"  Rotation     : {cam_obj.rotation_euler}\n\n")

        # World matrix (object-to-world, as Blender stores it)
        f.write("  matrix_world (4x4, object-to-world):\n")
        for row in cam_obj.matrix_world:
            f.write(f"    [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}]\n")
        f.write("\n")

        # Intrinsic matrix K
        K = get_intrinsic_matrix(cam_data, scene)
        f.write("  Intrinsic matrix K (3x3):\n")
        for row in K:
            f.write(f"    [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f}]\n")
        f.write("\n")

        # Extrinsic matrix (world-to-camera, OpenCV convention)
        E = get_extrinsic_matrix(cam_obj)
        f.write("  Extrinsic matrix (4x4, world-to-camera, OpenCV convention):\n")
        for row in E:
            f.write(f"    [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}]\n")
        f.write("\n")

        # Rotation and translation components
        R = E[:3, :3]
        t = E[:3, 3]
        f.write("  Rotation R (3x3):\n")
        for row in R:
            f.write(f"    [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}]\n")
        f.write(f"  Translation t: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n\n")

        # Full projection matrix P = K @ [R|t]
        P = get_projection_matrix(K, E)
        f.write("  Projection matrix P (3x4):\n")
        for row in P:
            f.write(f"    [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}]\n")
        f.write("\n")

        # Store for .npz export
        npz_data[cam_obj.name] = {"K": K, "R": R, "t": t, "P": P}

    # ─── Dart arrow vector (base → board) ────────────────────────────────────
    dart = bpy.data.objects.get('11750_throwing_dart_v1_L3')
    board = bpy.data.objects.get('11721_darboard_V4_L3')

    # Fallback: search by keyword
    if not dart or not board:
        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue
            name = obj.name.lower()
            if not board and 'board' in name:
                board = obj
            elif not dart and ('dart' in name or 'arrow' in name):
                dart = obj

    if dart and board:
        f.write(f"{'='*60}\n")
        f.write("Dart Arrow Vector (base -> board)\n\n")

        # Dart's long axis is local Z, tip at -Z end, base at +Z end
        bbox_world = [dart.matrix_world @ mathutils.Vector(c) for c in dart.bound_box]
        dart_forward_local = mathutils.Vector((0, 0, -1))
        dart_forward_world = (dart.matrix_world.to_3x3() @ dart_forward_local).normalized()

        local_z_axis = (dart.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))).normalized()
        origin = dart.matrix_world.translation.copy()

        # Project bbox corners onto the dart's local Z axis relative to origin
        z_projections = [(v - origin).dot(local_z_axis) for v in bbox_world]
        min_z = min(z_projections)  # tip end (-Z)
        max_z = max(z_projections)  # base end (+Z)

        # Reconstruct tip and base as points ON the dart's center axis
        tip_point = origin + local_z_axis * min_z
        base_point = origin + local_z_axis * max_z

        # Board front face position
        bbox_board = [board.matrix_world @ mathutils.Vector(c) for c in board.bound_box]
        board_front_y = min(v.y for v in bbox_board)
        board_center = board.location.copy()

        # Vector from dart base to board center
        base_to_board = board_center - base_point
        base_to_board_norm = base_to_board.normalized()

        f.write(f"  Dart object   : {dart.name}\n")
        f.write(f"  Board object  : {board.name}\n")
        f.write(f"  Dart location : ({dart.location.x:.6f}, {dart.location.y:.6f}, {dart.location.z:.6f})\n")
        f.write(f"  Dart rotation : ({dart.rotation_euler.x:.6f}, {dart.rotation_euler.y:.6f}, {dart.rotation_euler.z:.6f})\n\n")
        f.write(f"  Tip  position (world): ({tip_point.x:.6f}, {tip_point.y:.6f}, {tip_point.z:.6f})\n")
        f.write(f"  Base position (world): ({base_point.x:.6f}, {base_point.y:.6f}, {base_point.z:.6f})\n\n")
        f.write(f"  Dart forward direction (tip direction, local -Z in world):\n")
        f.write(f"    ({dart_forward_world.x:.6f}, {dart_forward_world.y:.6f}, {dart_forward_world.z:.6f})\n\n")
        f.write(f"  Board center  : ({board_center.x:.6f}, {board_center.y:.6f}, {board_center.z:.6f})\n")
        f.write(f"  Board front Y : {board_front_y:.6f}\n\n")
        f.write(f"  Vector base -> board center:\n")
        f.write(f"    ({base_to_board.x:.6f}, {base_to_board.y:.6f}, {base_to_board.z:.6f})\n")
        f.write(f"  Normalized:\n")
        f.write(f"    ({base_to_board_norm.x:.6f}, {base_to_board_norm.y:.6f}, {base_to_board_norm.z:.6f})\n")
        f.write(f"  Distance base -> board center: {base_to_board.length:.6f}\n\n")
    else:
        f.write(f"{'='*60}\n")
        f.write("WARNING: Could not find dart and/or board objects.\n")
        f.write(f"  Dart found: {dart is not None}\n")
        f.write(f"  Board found: {board is not None}\n\n")

# ─── Save machine-readable .npz ──────────────────────────────────────────────
cam_names = sorted(npz_data.keys())  # alphabetical: "Camera Left", "Camera Right"
if len(cam_names) >= 2:
    d1, d2 = npz_data[cam_names[0]], npz_data[cam_names[1]]
    npz_path = os.path.join(output_dir, "camera_data.npz")
    np.savez(
        npz_path,
        P1=d1["P"], P2=d2["P"],
        K=d1["K"],  # shared intrinsic
        R1=d1["R"], t1=d1["t"],
        R2=d2["R"], t2=d2["t"],
        cam1_name=cam_names[0],
        cam2_name=cam_names[1],
    )
    print(f"Camera data saved to: {npz_path}")

print(f"Camera matrices written to: {output_path}")