"""Tiny renderable fixture from a planning request. NOT the original textured room."""

import argparse
import json
import sys
from pathlib import Path

import bpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for i, box in enumerate([request["target"], *request["obstacles"]]):
        center = [(a + b) / 2 for a, b in zip(box["minimum"], box["maximum"])]
        extent = [b - a for a, b in zip(box["minimum"], box["maximum"])]
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        obj = bpy.context.object
        obj.name, obj.dimensions = box["object_id"], extent
        material = bpy.data.materials.new(f"proxy_material_{i}")
        material.diffuse_color = (0.16, 0.42, 0.6, 1.0) if i == 0 else (0.5, 0.5, 0.5, 1.0)
        obj.data.materials.append(material)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.world.color = (0.25, 0.25, 0.25)
    up = "XYZ".index(request["up_axis"])
    for sign in (-1, 1):
        position = [sign * 3.0, sign * 3.0, sign * 3.0]
        position[up] = 4.0
        bpy.ops.object.light_add(type="POINT", location=position)
        bpy.context.object.data.energy = 1000
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)


if __name__ == "__main__":
    main()
