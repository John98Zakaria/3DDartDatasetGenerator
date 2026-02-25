import bpy
import mathutils
import random
import math

# --- Find objects by name ---
board = bpy.data.objects.get('11721_darboard_V4_L3')
dart = bpy.data.objects.get('11750_throwing_dart_v1_L3')

if not board or not dart:
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        name = obj.name.lower()
        if 'board' in name:
            board = obj
        elif 'dart' in name or 'arrow' in name:
            dart = obj

print(f"Board: {board.name}")
print(f"Dart:  {dart.name}")

# --- Board measurements ---
# Board is at origin, thin along Y in world space
# Front face (facing the thrower at -Y) is at bbox min Y
bbox_board = [board.matrix_world @ mathutils.Vector(c) for c in board.bound_box]
board_front_y = min(v.y for v in bbox_board)  # ~ -0.67
board_center = board.location.copy()          # (0, 0, 0)
board_radius = max(board.dimensions.x, board.dimensions.z) / 2  # ~26.75

# --- Dart tip measurement ---
# Dart's long axis is Z, origin is very close to the tip (-Z end)
# Tip distance from origin = origin.z - bbox_min_z
bbox_dart = [dart.matrix_world @ mathutils.Vector(c) for c in dart.bound_box]
dart_tip_offset = dart.location.z - min(v.z for v in bbox_dart)  # ~0.44

print(f"Board front face Y: {board_front_y:.4f}")
print(f"Board radius: {board_radius:.2f}")
print(f"Dart tip offset from origin: {dart_tip_offset:.4f}")

# --- Random hit point on the board face (XZ plane) ---
angle = random.uniform(0, 2 * math.pi)
r = board_radius * math.sqrt(random.uniform(0, 0.85))
hit_x = board_center.x + r * math.cos(angle)
hit_z = board_center.z + r * math.sin(angle)

# The exact point on the board surface where the tip lands
tip_target = mathutils.Vector((hit_x, board_front_y, hit_z))

# --- Rotation: point the dart tip at the board ---
# Dart tip is along -Z in local space
# Board is in +Y direction from the thrower
# Align dart -Z local with +Y world
throw_dir = mathutils.Vector((0, 1, 0))
final_rot = throw_dir.to_track_quat('-Z', 'X').to_euler()

# --- Position at impact ---
# Place dart origin right at the board front face
hit_origin = mathutils.Vector((
    tip_target.x,
    -0.67,
    tip_target.z,
))

# --- Start position: far behind, approaching from -Y ---
start_origin = mathutils.Vector((
    hit_origin.x + random.uniform(-3, 3),
    hit_origin.y - 80,
    hit_origin.z + random.uniform(5, 15),
))

# Mid-air: arc peak
mid_origin = (start_origin + hit_origin) / 2
mid_origin.z += 5.0

# --- Rotation with wobble for start/mid ---
wobble_rot = mathutils.Euler((
    final_rot.x + math.radians(random.uniform(-10, 10)),
    final_rot.y + math.radians(random.uniform(-6, 6)),
    final_rot.z + math.radians(random.uniform(-6, 6)),
))
mid_rot = mathutils.Euler((
    final_rot.x + math.radians(random.uniform(-4, 4)),
    final_rot.y + math.radians(random.uniform(-2, 2)),
    final_rot.z + math.radians(random.uniform(-2, 2)),
))

# --- Clear old animation ---
dart.animation_data_clear()

# --- Keyframes ---
start_frame = 1
mid_frame = 15
hit_frame = 25
hold_frame = 50

bpy.context.scene.frame_start = start_frame
bpy.context.scene.frame_end = hold_frame

# Frame 1: Start
bpy.context.scene.frame_set(start_frame)
dart.location = start_origin
dart.rotation_euler = wobble_rot
dart.keyframe_insert(data_path="location", frame=start_frame)
dart.keyframe_insert(data_path="rotation_euler", frame=start_frame)

# Frame 15: Mid-air arc
bpy.context.scene.frame_set(mid_frame)
dart.location = mid_origin
dart.rotation_euler = mid_rot
dart.keyframe_insert(data_path="location", frame=mid_frame)
dart.keyframe_insert(data_path="rotation_euler", frame=mid_frame)

# Frame 25: Hit the board (tip on surface)
bpy.context.scene.frame_set(hit_frame)
dart.location = hit_origin
dart.rotation_euler = final_rot
dart.keyframe_insert(data_path="location", frame=hit_frame)
dart.keyframe_insert(data_path="rotation_euler", frame=hit_frame)

# Frame 50: Hold in place
bpy.context.scene.frame_set(hold_frame)
dart.location = hit_origin
dart.rotation_euler = final_rot
dart.keyframe_insert(data_path="location", frame=hold_frame)
dart.keyframe_insert(data_path="rotation_euler", frame=hold_frame)

# --- Easing ---
try:
    action = dart.animation_data.action
    if action is not None:
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                if kp.co[0] <= mid_frame:
                    kp.interpolation = 'BEZIER'
                    kp.easing = 'EASE_IN'
                elif kp.co[0] == hit_frame:
                    kp.interpolation = 'BEZIER'
                    kp.easing = 'EASE_OUT'
                else:
                    kp.interpolation = 'CONSTANT'
except Exception as e:
    print(f"Note: Easing not applied ({e}), animation still works.")

bpy.context.scene.frame_set(start_frame)

print(f"\nDart throw animated!")
print(f"  Tip lands at:    {tip_target}")
print(f"  Dart origin at:  {hit_origin}")
print(f"  Tip check:       origin.y ({hit_origin.y:.3f}) + offset ({dart_tip_offset:.3f}) = {hit_origin.y + dart_tip_offset:.3f} (board at {board_front_y:.3f})")
print(f"  Frames: {start_frame} -> {hit_frame} (hold until {hold_frame})")
