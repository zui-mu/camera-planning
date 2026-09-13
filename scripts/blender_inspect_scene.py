"""Read-only Blender scene audit for camera-planning inputs."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def open_source(source):
    suffix = source.suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(source))
    elif suffix in {".usd", ".usda", ".usdc", ".usdz"}:
        bpy.ops.wm.usd_import(filepath=str(source))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(source))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source))
    else:
        raise ValueError(f"unsupported scene format: {suffix}")


def bounds(obj, scale):
    points = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], dtype=float)
    points *= scale
    return points.min(axis=0).tolist(), points.max(axis=0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    source = Path(args.source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    open_source(source)
    scene = bpy.context.scene
    unit = scene.unit_settings
    scale = float(unit.scale_length) if unit.system == "METRIC" else 1.0
    records, all_bounds = [], []
    for obj in sorted(scene.objects, key=lambda item: item.name):
        record = {
            "name": obj.name,
            "type": obj.type,
            "hide_render": bool(obj.hide_render),
            "collections": sorted(collection.name for collection in obj.users_collection),
        }
        if obj.type == "MESH":
            low, high = bounds(obj, scale)
            all_bounds.extend((low, high))
            record.update(
                minimum_m=low,
                maximum_m=high,
                dimensions_m=(np.array(high) - np.array(low)).tolist(),
                vertices=len(obj.data.vertices),
                polygons=len(obj.data.polygons),
                materials=[slot.material.name for slot in obj.material_slots if slot.material],
            )
        records.append(record)
    images = []
    for image in bpy.data.images:
        if image.source not in {"FILE", "SEQUENCE", "MOVIE"}:
            continue
        resolved = Path(bpy.path.abspath(image.filepath)).resolve() if image.filepath else None
        images.append(
            {
                "name": image.name,
                "filepath": image.filepath,
                "resolved": str(resolved) if resolved else None,
                "exists": bool(image.packed_file or (resolved and resolved.is_file())),
                "packed": bool(image.packed_file),
            }
        )
    mesh_names = [item["name"] for item in records if item["type"] == "MESH"]
    keywords = {
        key: [name for name in mesh_names if key in name.lower()]
        for key in ("sofa", "chair", "table", "floor", "wall", "ceiling", "human", "body")
    }
    result = {
        "schema_version": "camera_scene_inspection_v1",
        "source": str(source),
        "blender_version": bpy.app.version_string,
        "unit_system": unit.system,
        "unit_scale_length": float(unit.scale_length),
        "interpreted_meters_per_unit": scale,
        "render_engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "object_count": len(records),
        "mesh_count": len(mesh_names),
        "camera_count": sum(item["type"] == "CAMERA" for item in records),
        "light_count": sum(item["type"] == "LIGHT" for item in records),
        "material_count": len(bpy.data.materials),
        "image_count": len(images),
        "missing_images": [image for image in images if not image["exists"]],
        "scene_minimum_m": np.min(all_bounds, axis=0).tolist() if all_bounds else None,
        "scene_maximum_m": np.max(all_bounds, axis=0).tolist() if all_bounds else None,
        "keyword_matches": keywords,
        "collections": sorted(collection.name for collection in bpy.data.collections),
        "objects": records,
    }
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
