"""Export evaluated world-space mesh geometry and a planner request.

The source .blend is read-only.  Target and solid-object names are explicit so
that a closed room shell is not accidentally interpreted as filled volume.
"""

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


def world_bbox(obj, meters_per_unit):
    points = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], dtype=float)
    points *= meters_per_unit
    return points.min(axis=0), points.max(axis=0)


def list_floats(value):
    return [float(x) for x in value.split(",")]


def box_mesh(low, high):
    vertices = np.array(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--exclude-object", action="append", default=[])
    parser.add_argument("--solid-object", action="append", default=[])
    parser.add_argument("--force-aabb-proxy", action="append", default=[])
    parser.add_argument("--large-object-proxy-threshold", type=int, default=250000)
    parser.add_argument("--target-max-triangles", type=int, default=60000)
    parser.add_argument("--allowed-region-object")
    parser.add_argument("--allowed-min", type=list_floats)
    parser.add_argument("--allowed-max", type=list_floats)
    parser.add_argument("--floor-height", type=float)
    parser.add_argument("--meters-per-unit", type=float)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fx", type=float, default=900.0)
    parser.add_argument("--fy", type=float, default=900.0)
    # Avoid --cx/--cy: Blender's Cycles add-on parses those as ambiguous
    # abbreviations of its own --cycles-* process arguments.
    parser.add_argument("--principal-x", type=float)
    parser.add_argument("--principal-y", type=float)
    parser.add_argument("--azimuth-count", type=int, default=16)
    parser.add_argument("--shell-offsets", type=list_floats, default=[2.0, 3.0])
    parser.add_argument("--camera-heights", type=list_floats, default=[1.2, 1.8])
    parser.add_argument("--focus-heights", type=list_floats, default=[1.0])
    parser.add_argument("--camera-clearance", type=float, default=0.12)
    parser.add_argument("--min-target-distance", type=float, default=1.5)
    parser.add_argument("--max-target-distance", type=float, default=6.0)
    parser.add_argument("--min-surface-visible-fraction", type=float, default=0.10)
    parser.add_argument("--min-envelope-visible-fraction", type=float, default=0.25)
    parser.add_argument("--surface-samples", type=int, default=360)
    parser.add_argument("--envelope-samples", type=int, default=360)
    parser.add_argument("--envelope-horizontal-clearance", type=float, default=0.80)
    parser.add_argument("--envelope-min-height", type=float, default=0.05)
    parser.add_argument("--full-body-height", type=float, default=2.10)
    parser.add_argument("--envelope-top-clearance", type=float, default=0.25)
    parser.add_argument("--image-margin-ratio", type=float, default=0.08)
    parser.add_argument("--underside-normal-threshold", type=float, default=-0.15)
    parser.add_argument("--observation-seed", type=int, default=17)
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--multiview-weight", type=float, default=1.0)
    parser.add_argument("--information-weight", type=float, default=0.3)
    parser.add_argument("--resolution-weight", type=float, default=0.3)
    parser.add_argument("--worst-zone-weight", type=float, default=0.5)
    parser.add_argument("--parallax-min-degrees", type=float, default=15.0)
    parser.add_argument("--pixel-noise-sigma", type=float, default=3.0)
    parser.add_argument("--information-scale", type=float, default=0.2)
    parser.add_argument("--target-pixels-per-meter", type=float, default=300.0)
    parser.add_argument("--swap-passes", type=int, default=2)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not bpy.data.filepath:
        raise ValueError("open a saved, frozen .blend before exporting geometry")
    if args.target_object not in bpy.data.objects:
        raise ValueError(f"target object not found: {args.target_object}")
    target = bpy.data.objects[args.target_object]
    if target.type != "MESH":
        raise ValueError("target object must be a mesh")
    excluded = set(args.exclude_object)
    if args.allowed_region_object:
        excluded.add(args.allowed_region_object)
    unknown = excluded.union(args.solid_object, args.force_aabb_proxy, {args.target_object}) - {
        obj.name for obj in bpy.data.objects
    }
    if unknown:
        raise ValueError(f"unknown Blender objects: {sorted(unknown)}")
    if args.meters_per_unit is not None:
        meters_per_unit = args.meters_per_unit
        unit_source = "explicit"
    elif bpy.context.scene.unit_settings.system == "METRIC":
        meters_per_unit = float(bpy.context.scene.unit_settings.scale_length)
        unit_source = "blender_metric_scale_length"
    else:
        meters_per_unit = 1.0
        unit_source = "assumed_one_meter_per_blender_unit"
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("meters per Blender unit must be positive")
    objects = sorted(
        (
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH" and obj.name not in excluded and not obj.hide_render
        ),
        key=lambda obj: obj.name,
    )
    if target not in objects:
        raise ValueError("target was excluded or hidden from geometry")
    dependencies = bpy.context.evaluated_depsgraph_get()
    vertices, faces, face_objects, object_records = [], [], [], []
    proxy_indices = []
    vertex_offset = 0
    for object_index, obj in enumerate(objects):
        planning_modifier = None
        source_vertex_count = None
        source_triangle_count = None
        if obj.name == args.target_object:
            source_evaluated = obj.evaluated_get(dependencies)
            source_mesh = source_evaluated.to_mesh(
                preserve_all_data_layers=False, depsgraph=dependencies
            )
            try:
                source_mesh.calc_loop_triangles()
                source_vertex_count = len(source_mesh.vertices)
                source_triangle_count = len(source_mesh.loop_triangles)
            finally:
                source_evaluated.to_mesh_clear()
            if source_triangle_count > args.target_max_triangles:
                planning_modifier = obj.modifiers.new(
                    name="__CameraPlanningTargetDecimate", type="DECIMATE"
                )
                planning_modifier.decimate_type = "COLLAPSE"
                planning_modifier.ratio = args.target_max_triangles / source_triangle_count
                planning_modifier.use_collapse_triangulate = True
                dependencies.update()
        evaluated = obj.evaluated_get(dependencies)
        mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=dependencies)
        try:
            mesh.calc_loop_triangles()
            world = evaluated.matrix_world
            original_vertex_count = (
                len(mesh.vertices) if source_vertex_count is None else source_vertex_count
            )
            original_triangle_count = (
                len(mesh.loop_triangles) if source_triangle_count is None else source_triangle_count
            )
            use_proxy = obj.name != args.target_object and (
                obj.name in args.force_aabb_proxy
                or original_triangle_count > args.large_object_proxy_threshold
            )
            if use_proxy:
                low, high = world_bbox(evaluated, meters_per_unit)
                object_vertices, object_faces = box_mesh(low, high)
                representation = "world_aabb_proxy"
                proxy_indices.append(object_index)
            else:
                object_vertices = np.array([world @ v.co for v in mesh.vertices], dtype=float)
                object_vertices *= meters_per_unit
                object_faces = np.array(
                    [triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int64
                )
                representation = (
                    "decimated_target_triangles"
                    if planning_modifier is not None
                    else "exact_evaluated_triangles"
                )
            if not len(object_vertices) or not len(object_faces):
                continue
            triangle_vertices = object_vertices[object_faces]
            area2 = np.linalg.norm(
                np.cross(
                    triangle_vertices[:, 1] - triangle_vertices[:, 0],
                    triangle_vertices[:, 2] - triangle_vertices[:, 0],
                ),
                axis=1,
            )
            nondegenerate = area2 >= 1e-14
            dropped_degenerate = int(np.count_nonzero(~nondegenerate))
            object_faces = object_faces[nondegenerate]
            if not len(object_faces):
                raise ValueError(f"object contains no nondegenerate triangles: {obj.name}")
            vertices.append(object_vertices)
            faces.append(object_faces + vertex_offset)
            face_objects.append(np.full(len(object_faces), object_index, dtype=np.int64))
            low, high = object_vertices.min(axis=0), object_vertices.max(axis=0)
            object_records.append(
                {
                    "object_index": object_index,
                    "object_id": obj.name,
                    "vertex_count": int(len(object_vertices)),
                    "triangle_count": int(len(object_faces)),
                    "source_vertex_count": int(original_vertex_count),
                    "source_triangle_count": int(original_triangle_count),
                    "representation": representation,
                    "dropped_degenerate_triangle_count": dropped_degenerate,
                    "minimum_m": low.tolist(),
                    "maximum_m": high.tolist(),
                }
            )
            vertex_offset += len(object_vertices)
        finally:
            evaluated.to_mesh_clear()
            if planning_modifier is not None:
                obj.modifiers.remove(planning_modifier)
                dependencies.update()
    if not vertices:
        raise ValueError("no render-visible mesh geometry found")
    vertices = np.concatenate(vertices)
    faces = np.concatenate(faces)
    face_objects = np.concatenate(face_objects)
    object_ids = np.array([record["object_id"] for record in object_records], dtype=np.str_)
    # object_records may omit empty objects; rebuild face indices if that ever occurs.
    if len(object_records) != len(objects):
        raise ValueError("empty evaluated mesh objects are unsupported; exclude them explicitly")
    solid_names = {args.target_object, *args.solid_object}
    solid_indices = np.array(
        sorted(
            set(proxy_indices).union(
                i for i, object_id in enumerate(object_ids) if object_id in solid_names
            )
        ),
        dtype=np.int64,
    )
    geometry_path = output / "scene_geometry.npz"
    np.savez_compressed(
        geometry_path,
        format_version=np.asarray("camera_scene_geometry_v1", dtype=np.str_),
        length_unit=np.asarray("meter", dtype=np.str_),
        vertices=vertices,
        faces=faces,
        face_object_indices=face_objects,
        object_ids=object_ids,
        solid_object_indices=solid_indices,
    )
    target_min, target_max = world_bbox(target, meters_per_unit)
    scene_min, scene_max = vertices.min(axis=0), vertices.max(axis=0)
    if args.allowed_region_object:
        allowed_min, allowed_max = world_bbox(
            bpy.data.objects[args.allowed_region_object], meters_per_unit
        )
        allowed_source = f"object:{args.allowed_region_object}"
    elif args.allowed_min is not None and args.allowed_max is not None:
        if len(args.allowed_min) != 3 or len(args.allowed_max) != 3:
            raise ValueError("allowed-min/max require three comma-separated meter values")
        allowed_min, allowed_max = np.array(args.allowed_min), np.array(args.allowed_max)
        allowed_source = "explicit_cli_meters"
    elif args.allowed_min is not None or args.allowed_max is not None:
        raise ValueError("pass both allowed-min and allowed-max")
    else:
        allowed_min, allowed_max = scene_min.copy(), scene_max.copy()
        allowed_min += 0.15
        allowed_max -= 0.15
        allowed_source = "scene_aabb_inset_UNVERIFIED"
    floor = float(scene_min[2] if args.floor_height is None else args.floor_height)
    request = {
        "schema_version": "camera_planning_request_v2",
        "scene_id": args.scene_id,
        "length_unit": "meter",
        "up_axis": "Z",
        "floor_height_m": floor,
        "budget": args.budget,
        "target": {
            "object_id": args.target_object,
            "minimum": target_min.tolist(),
            "maximum": target_max.tolist(),
        },
        "obstacles": [],
        "allowed_camera_region": {
            "object_id": "allowed_camera_region",
            "minimum": allowed_min.tolist(),
            "maximum": allowed_max.tolist(),
        },
        "intrinsics": {
            "width": args.width,
            "height": args.height,
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.width / 2 if args.principal_x is None else args.principal_x,
            "cy": args.height / 2 if args.principal_y is None else args.principal_y,
        },
        "candidates": {
            "azimuth_count": args.azimuth_count,
            "shell_offsets_m": args.shell_offsets,
            "heights_above_floor_m": args.camera_heights,
            "focus_heights_above_floor_m": args.focus_heights,
            "camera_clearance_m": args.camera_clearance,
            "min_target_distance_m": args.min_target_distance,
            "max_target_distance_m": args.max_target_distance,
            "min_surface_visible_fraction": args.min_surface_visible_fraction,
            "min_envelope_visible_fraction": args.min_envelope_visible_fraction,
        },
        "observation": {
            "mode": "furniture_surface_and_envelope",
            "surface_samples": args.surface_samples,
            "envelope_samples": args.envelope_samples,
            "horizontal_clearance_m": args.envelope_horizontal_clearance,
            "min_height_above_floor_m": args.envelope_min_height,
            "full_body_height_m": args.full_body_height,
            "top_clearance_m": args.envelope_top_clearance,
            "image_margin_ratio": args.image_margin_ratio,
            "underside_normal_threshold": args.underside_normal_threshold,
            "seed": args.observation_seed,
        },
        "scoring": {
            "coverage_weight": args.coverage_weight,
            "multiview_weight": args.multiview_weight,
            "information_weight": args.information_weight,
            "resolution_weight": args.resolution_weight,
            "worst_zone_weight": args.worst_zone_weight,
            "parallax_min_degrees": args.parallax_min_degrees,
            "pixel_noise_sigma": args.pixel_noise_sigma,
            "information_scale_m": args.information_scale,
            "target_pixels_per_meter": args.target_pixels_per_meter,
            "swap_passes": args.swap_passes,
        },
        "geometry": {
            "backend": "triangle_mesh",
            "mesh_file": str(geometry_path),
            "sha256": digest(geometry_path),
        },
    }
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    manifest = {
        "format_version": "camera_scene_export_v2",
        "scene_id": args.scene_id,
        "source_blend": bpy.data.filepath,
        "source_blend_sha256": digest(bpy.data.filepath),
        "blender_version": bpy.app.version_string,
        "length_unit": "meter",
        "meters_per_blender_unit": meters_per_unit,
        "unit_source": unit_source,
        "up_axis": "Z",
        "target_object": args.target_object,
        "solid_objects": sorted(solid_names),
        "force_aabb_proxy_objects": sorted(args.force_aabb_proxy),
        "large_object_proxy_threshold_triangles": args.large_object_proxy_threshold,
        "target_max_triangles": args.target_max_triangles,
        "excluded_objects": sorted(excluded),
        "allowed_region_source": allowed_source,
        "floor_height_source": "explicit"
        if args.floor_height is not None
        else "scene_min_UNVERIFIED",
        "geometry_sha256": digest(geometry_path),
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "objects": object_records,
        "proxy_object_count": len(proxy_indices),
        "warnings": [
            message
            for condition, message in (
                (
                    unit_source.startswith("assumed"),
                    "Unit scale was assumed; verify before research use.",
                ),
                (
                    "UNVERIFIED" in allowed_source,
                    "Allowed camera region was inferred from scene AABB.",
                ),
                (args.floor_height is None, "Floor height was inferred from minimum scene Z."),
                (
                    len(solid_names) == 1,
                    "Only the target is treated as a solid volume; declare other closed furniture explicitly.",
                ),
            )
            if condition
        ],
    }
    (output / "scene_export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
