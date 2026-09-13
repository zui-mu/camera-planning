"""Create a tiny metric room for engineering smoke tests, not research data."""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def make_material(name, color, roughness=0.55):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = roughness
    return material


def cube(name, location, scale, material, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel:
        modifier = obj.modifiers.new("SoftEdges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    return obj


def cylinder_between(name, start, end, radius, material):
    start, end = Vector(start), Vector(end)
    delta = end - start
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=20, radius=radius, depth=delta.length, location=(start + end) / 2
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(delta.normalized())
    obj.data.materials.append(material)
    return obj


def join_as(objects, name):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    objects[0].name = name
    return objects[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.world.color = (0.025, 0.03, 0.045)
    floor_mat = make_material("Floor", (0.23, 0.20, 0.17))
    wall_mat = make_material("Walls", (0.72, 0.74, 0.78))
    sofa_mat = make_material("SofaFabric", (0.15, 0.28, 0.38), 0.8)
    cabinet_mat = make_material("Cabinet", (0.32, 0.18, 0.10), 0.7)
    human_mat = make_material("ReferenceHuman", (0.72, 0.37, 0.28), 0.62)
    cube("Floor", (0, 0, -0.05), (4.5, 4.0, 0.05), floor_mat)
    cube("BackWall", (0, -4.0, 1.5), (4.5, 0.05, 1.5), wall_mat)
    cube("LeftWall", (-4.5, 0, 1.5), (0.05, 4.0, 1.5), wall_mat)
    cube("RightWall", (4.5, 0, 1.5), (0.05, 4.0, 1.5), wall_mat)
    sofa_parts = [
        cube("SofaSeat", (0, 0, 0.42), (1.3, 0.55, 0.18), sofa_mat, 0.08),
        cube("SofaBack", (0, 0.48, 0.93), (1.3, 0.16, 0.58), sofa_mat, 0.08),
        cube("SofaArmL", (-1.18, 0, 0.67), (0.18, 0.58, 0.42), sofa_mat, 0.07),
        cube("SofaArmR", (1.18, 0, 0.67), (0.18, 0.58, 0.42), sofa_mat, 0.07),
    ]
    sofa = join_as(sofa_parts, "Sofa")
    cube("SideCabinet", (2.45, 0.45, 0.55), (0.5, 0.45, 0.55), cabinet_mat, 0.03)
    # Deliberately simple seated mannequin. It validates acquisition plumbing, not SAM3D accuracy.
    human = [cube("Torso", (0, -0.02, 1.38), (0.28, 0.16, 0.42), human_mat, 0.1)]
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24, ring_count=16, radius=0.18, location=(0, -0.02, 1.94)
    )
    bpy.context.object.data.materials.append(human_mat)
    human.append(bpy.context.object)
    human += [
        cylinder_between("ArmL", (-0.26, -0.02, 1.56), (-0.48, -0.35, 1.12), 0.07, human_mat),
        cylinder_between("ArmR", (0.26, -0.02, 1.56), (0.48, -0.35, 1.12), 0.07, human_mat),
        cylinder_between("ThighL", (-0.14, -0.1, 1.08), (-0.18, -0.72, 0.78), 0.09, human_mat),
        cylinder_between("ThighR", (0.14, -0.1, 1.08), (0.18, -0.72, 0.78), 0.09, human_mat),
        cylinder_between("ShinL", (-0.18, -0.72, 0.78), (-0.18, -0.82, 0.18), 0.075, human_mat),
        cylinder_between("ShinR", (0.18, -0.72, 0.78), (0.18, -0.82, 0.18), 0.075, human_mat),
    ]
    join_as(human, "RefHuman")
    cube("CameraAllowedRegion", (0, 0, 1.5), (4.15, 3.65, 1.35), wall_mat)
    allowed = bpy.context.object
    allowed.display_type = "WIRE"
    allowed.hide_render = True
    for location, energy, size in (((0, 0, 4.5), 1500, 5.0), ((2, -2, 2.5), 700, 3.0)):
        data = bpy.data.lights.new("Area", "AREA")
        data.energy, data.shape, data.size = energy, "DISK", size
        light = bpy.data.objects.new("Area", data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (0, 0, math.pi)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(
        json.dumps(
            {
                "status": "saved",
                "scene": str(output),
                "target_object": sofa.name,
                "human_object": "RefHuman",
                "allowed_region_object": "CameraAllowedRegion",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
