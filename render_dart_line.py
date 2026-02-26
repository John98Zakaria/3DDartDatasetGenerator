import bpy
import mathutils
import numpy as np
import cv2
import os

OUTPUT_DIR = r"C:\Users\John\PycharmProjects\PythonProject"


def get_intrinsic_matrix(camera_data, scene):
    """Build the 3x3 intrinsic (calibration) matrix K."""
    render = scene.render
    res_x = render.resolution_x * render.resolution_percentage / 100
    res_y = render.resolution_y * render.resolution_percentage / 100

    focal_mm = camera_data.lens
    sensor_width = camera_data.sensor_width
    sensor_height = camera_data.sensor_height

    if camera_data.sensor_fit == 'VERTICAL':
        pixel_aspect = res_y / sensor_height
    else:
        pixel_aspect = res_x / sensor_width

    f_x = focal_mm * pixel_aspect
    f_y = focal_mm * pixel_aspect

    cx = res_x / 2.0
    cy = res_y / 2.0

    if camera_data.sensor_fit == 'VERTICAL':
        cx += camera_data.shift_x * res_y
        cy += camera_data.shift_y * res_y
    else:
        cx += camera_data.shift_x * res_x
        cy += camera_data.shift_y * res_x

    return np.array([
        [f_x,  0,   cx],
        [ 0,  f_y,  cy],
        [ 0,   0,    1],
    ])


def get_extrinsic_matrix(camera_obj):
    """Get the 4x4 world-to-camera extrinsic matrix (OpenCV convention).

    Uses normalized matrix_world to strip out any scale on the camera object.
    """
    # Normalize to remove scale, then invert to get world-to-camera
    mat = camera_obj.matrix_world.normalized().inverted()

    # Blender camera: -Z forward, Y up → OpenCV: Z forward, Y down
    flip = mathutils.Matrix((
        (1,  0,  0, 0),
        (0, -1,  0, 0),
        (0,  0, -1, 0),
        (0,  0,  0, 1),
    ))
    extrinsic = flip @ mat
    return np.array(extrinsic)


def project_point(P, world_point):
    """Project a 3D world point to 2D pixel coordinates using projection matrix P."""
    pt = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    projected = P @ pt
    px = projected[0] / projected[2]
    py = projected[1] / projected[2]
    return int(round(px)), int(round(py))


def get_dart_endpoints(dart):
    """Get tip and base positions along the dart's center axis in world space."""
    bbox_world = [dart.matrix_world @ mathutils.Vector(c) for c in dart.bound_box]
    local_z_axis = (dart.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))).normalized()
    origin = dart.matrix_world.translation.copy()

    z_projections = [(v - origin).dot(local_z_axis) for v in bbox_world]
    min_z = min(z_projections)  # tip end (-Z)
    max_z = max(z_projections)  # base end (+Z)

    tip_point = origin + local_z_axis * min_z
    base_point = origin + local_z_axis * max_z
    return tip_point, base_point


# ─── Find objects ─────────────────────────────────────────────────────────────
scene = bpy.context.scene
cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']

dart = bpy.data.objects.get('11750_throwing_dart_v1_L3')
if not dart:
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and ('dart' in obj.name.lower() or 'arrow' in obj.name.lower()):
            dart = obj
            break

if not dart:
    raise RuntimeError("Could not find dart object in scene")

tip_world, base_world = get_dart_endpoints(dart)
print(f"Dart tip  (world): {tip_world}")
print(f"Dart base (world): {base_world}")

# ─── Save original settings ──────────────────────────────────────────────────
original_camera = scene.camera
original_filepath = scene.render.filepath
original_format = scene.render.image_settings.file_format

scene.render.image_settings.file_format = 'PNG'

# ─── Render from each camera and draw the dart line ───────────────────────────
for cam_obj in cameras:
    cam_name = cam_obj.name.replace(" ", "_")

    # Render
    scene.camera = cam_obj
    render_path = os.path.join(OUTPUT_DIR, f"render_{cam_name}.png")
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {render_path}")

    # Build projection matrix P = K @ [R|t]
    K = get_intrinsic_matrix(cam_obj.data, scene)
    E = get_extrinsic_matrix(cam_obj)
    P = K @ E[:3, :]

    # Project dart endpoints to pixel coordinates
    tip_px = project_point(P, tip_world)
    base_px = project_point(P, base_world)
    print(f"  {cam_obj.name}: tip={tip_px}, base={base_px}")

    # Load rendered image and draw the line
    img = cv2.imread(render_path)
    cv2.line(img, tip_px, base_px, color=(0, 0, 255), thickness=3)
    cv2.circle(img, tip_px, 8, color=(0, 255, 0), thickness=-1)    # green = tip
    cv2.circle(img, base_px, 8, color=(255, 0, 0), thickness=-1)   # blue = base

    output_path = os.path.join(OUTPUT_DIR, f"dart_line_{cam_name}.png")
    cv2.imwrite(output_path, img)
    print(f"  Saved annotated image: {output_path}")

# ─── Restore original settings ───────────────────────────────────────────────
scene.camera = original_camera
scene.render.filepath = original_filepath
scene.render.image_settings.file_format = original_format

print("\nDone! Annotated images saved.")
