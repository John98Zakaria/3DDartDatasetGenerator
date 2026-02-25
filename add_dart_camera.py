import bpy
import mathutils

# Collect all objects that might be the dartboard or dart
dart_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

if dart_objects:
    # Calculate the center point of all dart-related objects
    total = mathutils.Vector((0.0, 0.0, 0.0))
    for obj in dart_objects:
        total += obj.location
    center = total / len(dart_objects)
else:
    center = mathutils.Vector((0.0, 0.0, 0.0))

# Position camera at an angle: offset in front and slightly to the side/above
camera_location = center + mathutils.Vector((3.0, -4.0, 2.0))

# Create the camera
bpy.ops.object.camera_add(location=camera_location)
camera = bpy.context.object
camera.name = "DartCamera"

# Point the camera at the center of the dart objects
direction = center - camera_location
rotation = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rotation.to_euler()
# Set focal length for a nice framing
camera.data.lens = 50

# Set as the active scene camera
bpy.context.scene.camera = camera

print(f"Camera added at {camera_location}, looking at {center}")
print(f"Objects in scene: {[obj.name for obj in dart_objects]}")
