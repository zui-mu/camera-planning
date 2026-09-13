"""Extract one target asset's world AABB from a standalone scene/model file."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def import_source(path):
    suffix = path.suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path))
    elif suffix in {".usd", ".usda", ".usdc", ".usdz"}:
        bpy.ops.wm.usd_import(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"unsupported target scene format: {suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-object")
    parser.add_argument("--exclude-object", action="append", default=[])
    parser.add_argument("--meters-per-unit", type=float)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    import_source(source)
    if args.meters_per_unit is not None:
        meters = args.meters_per_unit
        unit_source = "explicit"
    elif bpy.context.scene.unit_settings.system == "METRIC":
        meters = float(bpy.context.scene.unit_settings.scale_length)
        unit_source = "blender_metric_scale_length"
    else:
        meters = 1.0
        unit_source = "assumed_one_meter_per_blender_unit"
    excluded = set(args.exclude_object)
    meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name not in excluded and not obj.hide_render
    ]
    if args.target_object:
        matches = [obj for obj in meshes if obj.name == args.target_object]
        if len(matches) != 1:
            raise ValueError(f"target object must match exactly once: {args.target_object}")
        target = matches[0]
        selection = "explicit_object_name"
    else:
        if not meshes:
            raise ValueError("standalone target scene contains no eligible mesh")
        # A separately exported furniture file should contain one meaningful asset.
        # Taking the densest mesh avoids Blender's six-face startup Cube without
        # relying on furniture names. The ranking is recorded for manual audit.
        target = max(meshes, key=lambda obj: len(obj.data.polygons))
        selection = "largest_polygon_count"
    evaluated = target.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = np.array(
        [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box], dtype=float
    ) * meters
    # Upright PCA OBB measured from evaluated mesh vertices, not world AABB corners.
    # This is a reproducible orientation estimate, not minimum-volume OBB or semantic front.
    mesh = evaluated.to_mesh()
    try:
        local = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", local)
        world = local.reshape(-1, 3) @ np.asarray(evaluated.matrix_world)[:3, :3].T
        world = (world + np.asarray(evaluated.matrix_world)[:3, 3]) * meters
        if len(world) < 3 or not np.isfinite(world).all():
            raise ValueError("target mesh needs at least three finite vertices")
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(world[:, :2], rowvar=False))
        direction = eigenvectors[:, -1]
        if direction[np.argmax(abs(direction))] < 0:
            direction = -direction
        ambiguous = eigenvalues[-1] <= 1e-12 or eigenvalues[-1] / max(eigenvalues[0], 1e-12) < 1.05
        if ambiguous:
            direction = np.array([1.0, 0.0])
        axes = np.array([[direction[0], -direction[1], 0.0],
                         [direction[1], direction[0], 0.0], [0.0, 0.0, 1.0]])
        projected = world @ axes
        low, high = projected.min(axis=0), projected.max(axis=0)
        obb = {"center": (((low + high) / 2) @ axes.T).tolist(),
               "axes": axes.tolist(), "half_extents": np.maximum((high - low) / 2, 1e-5).tolist(),
               "method": "upright_mesh_pca_world_axis_fallback" if ambiguous else "upright_mesh_pca"}
    finally:
        evaluated.to_mesh_clear()
    candidates = sorted(
        (
            {
                "object_id": obj.name,
                "polygon_count": len(obj.data.polygons),
                "vertex_count": len(obj.data.vertices),
            }
            for obj in meshes
        ),
        key=lambda item: item["polygon_count"],
        reverse=True,
    )
    result = {
        "format_version": "camera_target_asset_bounds_v1",
        "source": str(source),
        "source_sha256": digest(source),
        "blender_version": bpy.app.version_string,
        "meters_per_blender_unit": meters,
        "unit_source": unit_source,
        "selection_policy": selection,
        "target_asset_object": target.name,
        "minimum_m": points.min(axis=0).tolist(),
        "maximum_m": points.max(axis=0).tolist(),
        "obb": obb,
        "obb_semantic_front_known": False,
        "eligible_mesh_ranking": candidates,
        "excluded_objects": sorted(excluded),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
