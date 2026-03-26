"""
Generate a YOLO pose dataset for dart tip/base keypoint detection.

Usage (headless):
    blender blender/dart-view.blend --background --python generate_yolo_dataset.py -- --num_samples 10 --output_dir ./dataset-multi-dart

Each sample places 0-3 darts (0 = negative sample ~5%), randomizes placement,
duplicates with colour variation, optionally adds noise, and renders from both
cameras, producing JPEG images and YOLO-pose label files.
"""

import bpy
import mathutils
import numpy as np
import math
import os
import sys
import json
import argparse
import time


# ── Argument parsing (after Blender's "--" separator) ────────────────────────

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Generate YOLO pose dataset (0-3 darts)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset-multi-dart"))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--render_width", type=int, default=640)
    parser.add_argument("--render_height", type=int, default=640)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args(argv)


# ── Projection helpers (from render_dart_line.py) ────────────────────────────

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
    """Get the 4x4 world-to-camera extrinsic matrix (OpenCV convention)."""
    mat = camera_obj.matrix_world.normalized().inverted()
    flip = mathutils.Matrix((
        (1,  0,  0, 0),
        (0, -1,  0, 0),
        (0,  0, -1, 0),
        (0,  0,  0, 1),
    ))
    extrinsic = flip @ mat
    return np.array(extrinsic)


def project_point(P, world_point):
    """Project a 3D world point to 2D pixel coords. Returns (px, py, depth)."""
    pt = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    projected = P @ pt
    depth = projected[2]
    if abs(depth) < 1e-7:
        return 0.0, 0.0, depth
    px = projected[0] / depth
    py = projected[1] / depth
    return px, py, depth


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


def get_dart_bbox_corners(dart):
    """Get all 8 bounding box corners in world space."""
    return [dart.matrix_world @ mathutils.Vector(c) for c in dart.bound_box]


# ── Dart randomization (from place_dart.py / throw_dart.py) ──────────────────

def randomize_dart(dart, board_radius, rng):
    """Place the dart at a random position/angle on the board."""
    # Clear any animation
    dart.animation_data_clear()
    bpy.context.scene.frame_set(1)

    # Random polar position on the board face
    angle = rng.uniform(0, 2 * math.pi)
    r = board_radius * math.sqrt(rng.uniform(0, 0.85))
    hit_x = r * math.cos(angle)
    hit_z = r * math.sin(angle)

    # Vary Y penetration depth slightly (board front is ~-0.67)
    y_depth = rng.uniform(-1.0, 0.5)
    dart.location = mathutils.Vector((hit_x, y_depth, hit_z))

    # Base rotation: dart tip (-Z local) points into board (+Y world)
    throw_dir = mathutils.Vector((0, 1, 0))
    base_rot = throw_dir.to_track_quat('-Z', 'X').to_euler()

    # Random tilt ±15° on X and Z
    base_rot.x += math.radians(rng.uniform(-15, 15))
    base_rot.z += math.radians(rng.uniform(-15, 15))
    dart.rotation_euler = base_rot

    # Force Blender to update transforms
    bpy.context.view_layer.update()


def randomize_hdri_rotation(world, rng):
    """Rotate the HDRI environment map randomly for lighting diversity."""
    if world is None or not world.use_nodes:
        return

    node_tree = world.node_tree
    for node in node_tree.nodes:
        if node.type == 'MAPPING':
            node.inputs['Rotation'].default_value[2] = rng.uniform(0, 2 * math.pi)
            return

    # If no Mapping node exists, look for Environment Texture and add rotation
    for node in node_tree.nodes:
        if node.type == 'TEX_COORD':
            return  # Already has coordinate setup, skip


# ── Dart duplication & colour randomization ──────────────────────────────────

def set_dart_visibility(dart, dart_children, visible):
    """Show or hide the dart and all its children."""
    dart.hide_render = not visible
    dart.hide_viewport = not visible
    for child in dart_children:
        child.hide_render = not visible
        child.hide_viewport = not visible


def duplicate_dart_hierarchy(original_dart, dart_children, rng):
    """Duplicate the dart and all its children, maintaining parent-child relationships."""
    # Copy parent
    new_parent = original_dart.copy()
    if original_dart.data:
        new_parent.data = original_dart.data.copy()
    bpy.context.collection.objects.link(new_parent)

    # Copy children and re-parent them
    new_children = []
    for child in dart_children:
        new_child = child.copy()
        if child.data:
            new_child.data = child.data.copy()
        bpy.context.collection.objects.link(new_child)
        new_child.parent = new_parent
        new_children.append(new_child)

    # Randomize material colours on all parts
    randomize_dart_colour(new_parent, rng)
    for child in new_children:
        randomize_dart_colour(child, rng)

    return new_parent, new_children


def randomize_dart_colour(dart, rng):
    """Randomize the diffuse/base colour of all materials on a dart."""
    for slot in dart.material_slots:
        if slot.material is None:
            continue
        # Make a unique copy so we don't affect the original
        slot.material = slot.material.copy()
        mat = slot.material
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                base_color = node.inputs.get('Base Color')
                if base_color and hasattr(base_color, 'default_value'):
                    r, g, b = rng.uniform(0.05, 1.0, 3)
                    base_color.default_value = (r, g, b, 1.0)


def cleanup_duplicates(duplicates):
    """Remove duplicated dart objects (parents + children) and their unique materials."""
    all_objs = []
    for item in duplicates:
        if isinstance(item, tuple):
            parent, children = item
            all_objs.extend(children)
            all_objs.append(parent)
        else:
            all_objs.append(item)

    for obj in all_objs:
        mats_to_remove = []
        for slot in obj.material_slots:
            if slot.material:
                mats_to_remove.append(slot.material)
        bpy.data.objects.remove(obj, do_unlink=True)
        for mat in mats_to_remove:
            if mat.users == 0:
                bpy.data.materials.remove(mat)


# ── Post-render noise ────────────────────────────────────────────────────────

def add_noise_to_image(img_path, rng, noise_type='gaussian', intensity=0.05):
    """Add noise to a rendered JPEG image using raw file manipulation.

    noise_type: 'gaussian', 'salt_pepper', or 'brightness'
    """
    # Read JPEG via Blender's image API
    img = bpy.data.images.load(img_path)
    pixels = np.array(img.pixels[:]).reshape(-1, 4)  # RGBA float [0,1]

    if noise_type == 'gaussian':
        noise = rng.normal(0, intensity, pixels[:, :3].shape)
        pixels[:, :3] = np.clip(pixels[:, :3] + noise, 0.0, 1.0)
    elif noise_type == 'salt_pepper':
        mask = rng.random(len(pixels))
        salt = mask < intensity / 2
        pepper = mask > (1 - intensity / 2)
        pixels[salt, :3] = 1.0
        pixels[pepper, :3] = 0.0
    elif noise_type == 'brightness':
        factor = rng.uniform(0.7, 1.3)
        pixels[:, :3] = np.clip(pixels[:, :3] * factor, 0.0, 1.0)

    img.pixels = pixels.flatten().tolist()
    img.save_render(img_path)
    bpy.data.images.remove(img)


# ── Annotation computation ───────────────────────────────────────────────────

def keypoint_visibility(px, py, depth, res_x, res_y):
    """Determine YOLO keypoint visibility flag.

    Returns:
        2 = visible (in frame, in front of camera)
        1 = labeled but out of frame
        0 = behind camera (invalid)
    """
    if depth <= 0:
        return 0
    if 0 <= px <= res_x and 0 <= py <= res_y:
        return 2
    return 1


def compute_yolo_annotation(dart, cam_obj, scene, res_x, res_y):
    """Compute YOLO pose annotation for one camera view.

    Returns:
        annotation string "0 cx cy w h tip_x tip_y tip_v base_x base_y base_v"
        or None if the dart is entirely behind the camera.
    """
    K = get_intrinsic_matrix(cam_obj.data, scene)
    E = get_extrinsic_matrix(cam_obj)
    P = K @ E[:3, :]

    # -- Keypoints: tip and base --
    tip_world, base_world = get_dart_endpoints(dart)
    tip_px, tip_py, tip_depth = project_point(P, tip_world)
    base_px, base_py, base_depth = project_point(P, base_world)

    tip_v = keypoint_visibility(tip_px, tip_py, tip_depth, res_x, res_y)
    base_v = keypoint_visibility(base_px, base_py, base_depth, res_x, res_y)

    # Skip if both keypoints are behind the camera
    if tip_v == 0 and base_v == 0:
        return None

    # -- Bounding box from all 8 dart bbox corners --
    bbox_corners = get_dart_bbox_corners(dart)
    projected_xs = []
    projected_ys = []

    for corner in bbox_corners:
        cx_px, cy_px, d = project_point(P, corner)
        if d > 0:  # Only use corners in front of camera
            projected_xs.append(cx_px)
            projected_ys.append(cy_px)

    if len(projected_xs) < 2:
        return None  # Not enough visible corners for a bbox

    # Clamp bbox to image bounds
    x_min = max(0, min(projected_xs))
    x_max = min(res_x, max(projected_xs))
    y_min = max(0, min(projected_ys))
    y_max = min(res_y, max(projected_ys))

    if x_max <= x_min or y_max <= y_min:
        return None  # Degenerate bbox

    # Normalize to [0, 1]
    cx = ((x_min + x_max) / 2.0) / res_x
    cy = ((y_min + y_max) / 2.0) / res_y
    w = (x_max - x_min) / res_x
    h = (y_max - y_min) / res_y

    # Clamp normalized keypoints (keep original for visibility, but normalize coords)
    norm_tip_x = tip_px / res_x
    norm_tip_y = tip_py / res_y
    norm_base_x = base_px / res_x
    norm_base_y = base_py / res_y

    # Format: class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
    return (f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
            f"{norm_tip_x:.6f} {norm_tip_y:.6f} {tip_v} "
            f"{norm_base_x:.6f} {norm_base_y:.6f} {base_v}")


# ── Render setup ─────────────────────────────────────────────────────────────

def setup_render(scene, width, height):
    """Configure EEVEE rendering for speed."""
    # Blender 4.0+: BLENDER_EEVEE_NEXT, older: BLENDER_EEVEE
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE'

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 90

    # Low TAA samples for fast rendering
    if hasattr(scene.eevee, 'taa_render_samples'):
        scene.eevee.taa_render_samples = 16

    # Disable compositing/sequencer for speed
    scene.render.use_compositing = False
    scene.render.use_sequencer = False


# ── Output directory setup ───────────────────────────────────────────────────

def setup_output_dirs(output_dir):
    """Create dataset-solo-dart directory structure."""
    dirs = {
        "train_images": os.path.join(output_dir, "images", "train"),
        "val_images": os.path.join(output_dir, "images", "val"),
        "train_labels": os.path.join(output_dir, "labels", "train"),
        "val_labels": os.path.join(output_dir, "labels", "val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def write_dataset_yaml(output_dir):
    """Write dataset-solo-dart.yaml for Ultralytics YOLO training."""
    abs_path = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""# YOLO Pose Dataset - Dart Tip & Base Keypoints
# Auto-generated by generate_yolo_dataset.py

path: {abs_path}
train: images/train
val: images/val

# Class names
names:
  0: dart

# Keypoints
kpt_shape: [2, 3]  # 2 keypoints, 3 values each (x, y, visibility)
"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Wrote {yaml_path}")


# ── Progress tracking ────────────────────────────────────────────────────────

def load_progress(output_dir):
    """Load progress file for resume support."""
    progress_path = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            return json.load(f)
    return {"completed_indices": []}


def save_progress(output_dir, progress):
    """Save progress file."""
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path, "w") as f:
        json.dump(progress, f)


# ── Main generation loop ────────────────────────────────────────────────────

def main():
    args = parse_args()

    scene = bpy.context.scene

    # Find objects
    dart = bpy.data.objects.get('11750_throwing_dart_v1_L3')
    if not dart:
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and ('dart' in obj.name.lower() or 'arrow' in obj.name.lower()):
                dart = obj
                break
    if not dart:
        raise RuntimeError("Could not find dart object in scene")

    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if not cameras:
        raise RuntimeError("No cameras found in scene")

    board = bpy.data.objects.get('11721_darboard_V4_L3')
    board_radius = 26.0
    if board:
        board_radius = max(board.dimensions.x, board.dimensions.z) / 2

    world = scene.world

    # Collect dart and all its children (the model may be a hierarchy)
    def get_all_children(obj):
        children = []
        for child in obj.children:
            children.append(child)
            children.extend(get_all_children(child))
        return children

    dart_children = get_all_children(dart)

    print(f"Found dart: {dart.name}")
    if dart_children:
        print(f"  Dart children: {[c.name for c in dart_children]}")
    print(f"Found {len(cameras)} cameras: {[c.name for c in cameras]}")
    print(f"Board radius: {board_radius:.1f}")
    print(f"All scene objects: {[o.name for o in bpy.data.objects]}")

    # Setup
    setup_render(scene, args.render_width, args.render_height)
    dirs = setup_output_dirs(args.output_dir)
    write_dataset_yaml(args.output_dir)

    res_x = args.render_width
    res_y = args.render_height

    # Resume support
    progress = load_progress(args.output_dir) if args.resume else {"completed_indices": []}
    completed_set = set(progress["completed_indices"])

    # Determine train/val split threshold
    num_val = int(args.num_samples * args.val_ratio)
    val_start = args.num_samples - num_val  # Last N samples are val

    total_images = 0
    skipped = 0
    failed = 0
    negatives = 0
    start_time = time.time()

    # Dart count distribution: ~5% with 0, ~35% with 1, ~35% with 2, ~25% with 3
    dart_count_weights = [0.10, 0.30, 0.35, 0.25]
    dart_count_choices = [0, 1, 2, 3]

    # Noise probability (~30% of images get some noise)
    noise_prob = 0.30
    noise_types = ['gaussian', 'salt_pepper', 'brightness']

    print(f"\nGenerating {args.num_samples} samples ({args.num_samples - num_val} train, {num_val} val)")
    print(f"Rendering at {res_x}x{res_y} with EEVEE, 16 TAA samples")
    print(f"Dart count distribution: {dict(zip(dart_count_choices, dart_count_weights))}")
    print(f"Noise probability: {noise_prob*100:.0f}%")
    print(f"Start index: {args.start_index}")
    if args.resume:
        print(f"Resuming: {len(completed_set)} samples already completed")
    print()

    for i in range(args.start_index, args.start_index + args.num_samples):
        # Skip if already completed (resume mode)
        if i in completed_set:
            skipped += 1
            continue

        # Deterministic seed per sample
        sample_seed = args.seed + i
        rng = np.random.RandomState(sample_seed)

        # Determine split
        relative_idx = i - args.start_index
        if relative_idx >= val_start:
            split = "val"
            img_dir = dirs["val_images"]
            lbl_dir = dirs["val_labels"]
        else:
            split = "train"
            img_dir = dirs["train_images"]
            lbl_dir = dirs["train_labels"]

        # Decide how many darts for this sample (0-3)
        num_darts = rng.choice(dart_count_choices, p=dart_count_weights)

        # Decide if this sample gets noise
        apply_noise = rng.random() < noise_prob
        noise_type = rng.choice(noise_types) if apply_noise else None
        noise_intensity = rng.uniform(0.02, 0.08) if apply_noise else 0

        # Randomize HDRI lighting
        randomize_hdri_rotation(world, rng)

        # Build list of active darts (original + duplicates)
        active_darts = []
        duplicates = []

        if num_darts == 0:
            # Negative sample: hide the original dart and all children
            set_dart_visibility(dart, dart_children, False)
            negatives += 1
        else:
            # Show original dart and all children
            set_dart_visibility(dart, dart_children, True)
            randomize_dart(dart, board_radius, rng)
            active_darts.append(dart)

            # Duplicate additional darts (num_darts - 1 copies)
            for _ in range(num_darts - 1):
                dup_parent, dup_children = duplicate_dart_hierarchy(dart, dart_children, rng)
                randomize_dart(dup_parent, board_radius, rng)
                duplicates.append((dup_parent, dup_children))
                active_darts.append(dup_parent)

        sample_ok = True

        # Render from each camera
        for cam_obj in cameras:
            cam_name = cam_obj.name.replace(" ", "_")
            filename = f"{i:06d}_{cam_name}"

            img_path = os.path.join(img_dir, f"{filename}.jpg")
            # Blender render filepath needs path without extension (it adds its own)
            render_path = os.path.join(img_dir, filename)
            lbl_path = os.path.join(lbl_dir, f"{filename}.txt")

            # Compute annotations for all active darts
            annotations = []
            for d in active_darts:
                ann = compute_yolo_annotation(d, cam_obj, scene, res_x, res_y)
                if ann is not None:
                    annotations.append(ann)

            # For non-negative samples, skip if no valid annotations at all
            if num_darts > 0 and len(annotations) == 0:
                sample_ok = False
                continue

            # Render
            scene.camera = cam_obj
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)

            # Blender may save as .jpg - find the actual file
            if not os.path.exists(img_path):
                # Check common Blender naming patterns
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = render_path + ext
                    if os.path.exists(candidate) and candidate != img_path:
                        os.rename(candidate, img_path)
                        break

            # Add noise if selected
            if apply_noise and os.path.exists(img_path):
                try:
                    add_noise_to_image(img_path, rng, noise_type, noise_intensity)
                except Exception as e:
                    print(f"  Warning: noise failed for {filename}: {e}")

            # Write label (empty file for negative samples)
            with open(lbl_path, "w") as f:
                for ann in annotations:
                    f.write(ann + "\n")

            total_images += 1

        # Cleanup duplicated darts
        if duplicates:
            cleanup_duplicates(duplicates)

        # Restore original dart visibility
        set_dart_visibility(dart, dart_children, True)

        if not sample_ok:
            failed += 1

        # Track progress
        progress["completed_indices"].append(i)

        # Save progress periodically (every 100 samples)
        if (relative_idx + 1) % 100 == 0:
            save_progress(args.output_dir, progress)

        # Print progress
        done = relative_idx + 1 - skipped
        if done > 0 and done % 10 == 0:
            elapsed = time.time() - start_time
            rate = done / elapsed
            remaining = (args.num_samples - done - skipped) / rate if rate > 0 else 0
            print(f"[{done}/{args.num_samples}] {total_images} images, "
                  f"{failed} failed, {negatives} negatives, {rate:.1f} samples/s, "
                  f"~{remaining/60:.0f}min remaining ({split})")

    # Final save
    save_progress(args.output_dir, progress)

    elapsed = time.time() - start_time
    print(f"\nDone! Generated {total_images} images in {elapsed/60:.1f} minutes")
    print(f"  Skipped (resume): {skipped}")
    print(f"  Failed (no valid annotation): {failed}")
    print(f"  Negative samples (0 darts): {negatives}")
    print(f"  Output: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
