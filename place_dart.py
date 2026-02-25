import bpy
import mathutils
import random
import math

# --- Find objects ---
board = bpy.data.objects.get('11721_darboard_V4_L3')
dart = bpy.data.objects.get('11750_throwing_dart_v1_L3')

# Clear any existing animation
dart.animation_data_clear()
bpy.context.scene.frame_set(1)

# --- Board info ---
# Front face at Y = -0.67, back at Y = 1.54
# Board spans X: -26.75 to 26.75, Z: -26.75 to 26.75
board_radius = 26.0

# --- Random spot on the board ---
angle = random.uniform(0, 2 * math.pi)
r = board_radius * math.sqrt(random.uniform(0, 0.85))
hit_x = r * math.cos(angle)
hit_z = r * math.sin(angle)

# --- Place dart ---
# Try Y=0 (board center), tip should be on the board
dart.location = mathutils.Vector((hit_x, 0, hit_z))

# Rotate so dart points into the board (+Y direction)
# Dart long axis is local Z, tip at -Z end
# Align -Z with +Y (into the board), then add random tilt
throw_dir = mathutils.Vector((0, 1, 0))
base_rot = throw_dir.to_track_quat('-Z', 'X').to_euler()

# Random angle: tilt the dart slightly off-center (like a real throw)
base_rot.x += math.radians(random.uniform(-15, 15))
base_rot.z += math.radians(random.uniform(-15, 15))
dart.rotation_euler = base_rot

print(f"Dart placed at: {dart.location}")
print(f"Dart rotation: {dart.rotation_euler}")
print(f"Hit spot: X={hit_x:.2f}, Z={hit_z:.2f}")
print()
print("If the tip is NOT on the board, try adjusting Y:")
print("  - Move Y more positive to push dart deeper into board")
print("  - Move Y more negative to pull dart away from board")
