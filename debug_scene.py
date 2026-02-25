import bpy
import mathutils

print("=== ALL OBJECTS IN SCENE ===")
for obj in bpy.data.objects:
    print(f"  Name: {obj.name}")
    print(f"    Type: {obj.type}")
    print(f"    Location: {obj.location}")
    print(f"    Rotation: {obj.rotation_euler}")
    print(f"    Dimensions: {obj.dimensions}")
    if obj.type == 'MESH':
        # Bounding box corners in world space
        bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        min_corner = mathutils.Vector((
            min(v.x for v in bbox),
            min(v.y for v in bbox),
            min(v.z for v in bbox),
        ))
        max_corner = mathutils.Vector((
            max(v.x for v in bbox),
            max(v.y for v in bbox),
            max(v.z for v in bbox),
        ))
        print(f"    BBox world min: {min_corner}")
        print(f"    BBox world max: {max_corner}")
        print(f"    Origin offset from center: {obj.location - (min_corner + max_corner) / 2}")
    print()
