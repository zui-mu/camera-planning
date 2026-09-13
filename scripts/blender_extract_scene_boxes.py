"""Extract lightweight world-space AABBs from a frozen Blender scene.

This script never edits or saves the source scene. A separately exported target
asset can identify the target by its world-space bounds. Room-shell AABBs can be
discarded by geometric relation to the explicit allowed camera region, without
depending on English object names.
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


def padded_box(low, high, minimum_thickness):
    low, high = np.asarray(low, dtype=float).copy(), np.asarray(high, dtype=float).copy()
    thin = high - low < minimum_thickness
    center = (low + high) / 2
    low[thin] = center[thin] - minimum_thickness / 2
    high[thin] = center[thin] + minimum_thickness / 2
    return low, high, np.flatnonzero(thin).tolist()


def automatic_structure(low, high, allowed_low, allowed_high):
    """Identify AABBs that are proxies for the room shell, not solid obstacles."""
    extent = high - low
    allowed_extent = allowed_high - allowed_low
    center = (allowed_low + allowed_high) / 2
    contains_center = np.all((center >= low) & (center <= high))
    coverage = extent / np.maximum(allowed_extent, 1e-9)
    # A single wall/exterior mesh often has one AABB enclosing the entire room.
    enclosing_shell = contains_center and np.count_nonzero(coverage >= 0.85) >= 2
    # Floors, ceilings, rugs and room-wide support slabs are irrelevant to a
    # camera whose allowed z-range is explicit; their full AABB must not fill it.
    horizontal_slab = coverage[0] >= 0.75 and coverage[1] >= 0.75 and extent[2] <= 0.30
    return enclosing_shell or horizontal_slab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-object")
    parser.add_argument("--target-bounds")
    parser.add_argument("--allowed-region-object")
    parser.add_argument("--allowed-min")
    parser.add_argument("--allowed-max")
    parser.add_argument("--exclude-object", action="append", default=[])
    parser.add_argument("--ignore-name-token", action="append", default=[])
    parser.add_argument(
        "--structure-policy",
        choices=["auto_geometry", "name_tokens", "none"],
        default="auto_geometry",
    )
    parser.add_argument("--meters-per-unit", type=float)
    parser.add_argument("--minimum-thickness", type=float, default=0.002)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    if not bpy.data.filepath:
        raise ValueError("open a saved, frozen .blend before extracting boxes")
    if not args.target_object and not args.target_bounds:
        raise ValueError("provide --target-object or --target-bounds")
    if args.allowed_region_object and args.allowed_region_object not in bpy.data.objects:
        raise ValueError(f"allowed region object not found: {args.allowed_region_object}")
    if args.minimum_thickness <= 0:
        raise ValueError("minimum thickness must be positive")

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

    if args.allowed_region_object:
        allowed_obj = bpy.data.objects[args.allowed_region_object]
        allowed_low, allowed_high = world_bbox(
            allowed_obj.evaluated_get(bpy.context.evaluated_depsgraph_get()), meters_per_unit
        )
    elif args.allowed_min and args.allowed_max:
        allowed_low = np.fromstring(args.allowed_min, sep=",")
        allowed_high = np.fromstring(args.allowed_max, sep=",")
        if allowed_low.shape != (3,) or allowed_high.shape != (3,):
            raise ValueError("allowed min/max must contain three comma-separated values")
    else:
        allowed_low = allowed_high = None
    if args.structure_policy == "auto_geometry" and allowed_low is None:
        raise ValueError("auto_geometry structure policy requires an allowed region")

    target_asset = None
    if args.target_bounds:
        target_asset = json.loads(Path(args.target_bounds).read_text(encoding="utf-8"))
        match_low = np.asarray(target_asset["minimum_m"], dtype=float)
        match_high = np.asarray(target_asset["maximum_m"], dtype=float)
        candidates = []
        excluded_for_match = set(args.exclude_object)
        dependencies = bpy.context.evaluated_depsgraph_get()
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH" or obj.hide_render or obj.name in excluded_for_match:
                continue
            low, high = world_bbox(obj.evaluated_get(dependencies), meters_per_unit)
            error = max(
                float(np.max(np.abs(low - match_low))),
                float(np.max(np.abs(high - match_high))),
            )
            candidates.append((error, obj))
        candidates.sort(key=lambda item: item[0])
        matches = [item for item in candidates if item[0] <= 1e-4]
        if len(matches) != 1:
            raise ValueError(
                "standalone target bounds must match exactly one full-scene mesh within "
                f"0.1 mm; found {len(matches)}"
            )
        matched_name = matches[0][1].name
        if args.target_object and args.target_object != matched_name:
            raise ValueError(
                f"explicit target {args.target_object} disagrees with bounds match {matched_name}"
            )
        args.target_object = matched_name
        target_match_error_m = matches[0][0]
    else:
        target_match_error_m = None
    if args.target_object not in bpy.data.objects:
        raise ValueError(f"target object not found: {args.target_object}")
    target = bpy.data.objects[args.target_object]
    if target.type != "MESH":
        raise ValueError("target object must be a mesh")

    excluded = set(args.exclude_object)
    tokens = tuple(token.casefold() for token in args.ignore_name_token if token)
    dependencies = bpy.context.evaluated_depsgraph_get()
    records = []
    ignored = []
    protected = {args.target_object, args.allowed_region_object}
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        if obj.type != "MESH" or obj.name in excluded:
            continue
        if obj.hide_render and obj.name not in protected:
            continue
        if (
            obj.name not in protected
            and args.structure_policy == "name_tokens"
            and tokens
            and any(token in obj.name.casefold() for token in tokens)
        ):
            ignored.append({"object_id": obj.name, "reason": "structural_name_token"})
            continue
        evaluated = obj.evaluated_get(dependencies)
        low, high = world_bbox(evaluated, meters_per_unit)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            ignored.append({"object_id": obj.name, "reason": "nonfinite_bounds"})
            continue
        low, high, padded_axes = padded_box(low, high, args.minimum_thickness)
        if (
            obj.name not in protected
            and args.structure_policy == "auto_geometry"
            and automatic_structure(low, high, allowed_low, allowed_high)
        ):
            ignored.append({"object_id": obj.name, "reason": "automatic_room_shell_geometry"})
            continue
        records.append(
            {
                "object_id": obj.name,
                "minimum_m": low.tolist(),
                "maximum_m": high.tolist(),
                "padded_axes": padded_axes,
                "role": (
                    "target"
                    if obj.name == args.target_object
                    else "allowed_region"
                    if obj.name == args.allowed_region_object
                    else "obstacle"
                ),
            }
        )
    if not any(record["role"] == "target" for record in records):
        raise ValueError("target was excluded from extracted boxes")

    result = {
        "format_version": "camera_scene_boxes_v1",
        "source_blend": bpy.data.filepath,
        "source_blend_sha256": digest(bpy.data.filepath),
        "blender_version": bpy.app.version_string,
        "length_unit": "meter",
        "up_axis": "Z",
        "meters_per_blender_unit": meters_per_unit,
        "unit_source": unit_source,
        "target_object": args.target_object,
        "target_asset": target_asset,
        "target_match_error_m": target_match_error_m,
        "boxes": records,
        "ignored": ignored,
        "excluded_objects": sorted(excluded),
        "ignored_name_tokens": list(tokens),
        "structure_policy": args.structure_policy,
        "warning": (
            "AABBs are conservative proxies. Concave objects, rotated furniture and gaps "
            "inside or beneath objects are represented as occupied box volume."
        ),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
