"""Run inside Blender. Renders a frozen .blend snapshot; never saves/edits the source file.

This renderer currently supports centered principal point, zero skew and fx==fy only.
Unsupported intrinsics fail rather than silently producing miscalibrated images.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure(scene, obj, camera):
    k = camera["K"]
    width, height = camera["width"], camera["height"]
    if camera["coordinate_convention"] != "world_to_camera":
        raise ValueError("renderer requires world_to_camera")
    if (
        abs(k[0][0] - k[1][1]) > 1e-6
        or abs(k[0][2] - width / 2) > 1e-6
        or abs(k[1][2] - height / 2) > 1e-6
        or abs(k[0][1]) > 1e-8
        or abs(k[1][0]) > 1e-8
        or k[2] != [0, 0, 1]
    ):
        raise ValueError("renderer supports centered, zero-skew, square-pixel K only")
    rotation = Matrix(camera["R"])
    center = -(rotation.transposed() @ Vector(camera["T"]))
    world = (rotation.transposed() @ Matrix.Diagonal((1.0, -1.0, -1.0))).to_4x4()
    world.translation = center
    obj.matrix_world = world
    obj.data.type = "PERSP"
    obj.data.sensor_fit = "HORIZONTAL"
    obj.data.sensor_width = 36.0
    obj.data.lens = k[0][0] * 36.0 / width
    obj.data.shift_x = obj.data.shift_y = 0.0
    obj.data.clip_start, obj.data.clip_end = 0.01, 1000.0
    scene.camera = obj
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.use_border = False
    bpy.context.view_layer.update()
    max_error = 0.0
    # This validates off-axis projections as well as the central ray.
    for local_point in ([0.0, 0.0, 2.0], [0.3, 0.2, 3.0], [-0.2, 0.4, 4.0]):
        p = Vector(local_point)
        world_point = rotation.transposed() @ (p - Vector(camera["T"]))
        ndc = world_to_camera_view(scene, obj, world_point)
        uv_blender = Vector((ndc.x * width, (1 - ndc.y) * height))
        projected = Matrix(k) @ p
        uv_cv = Vector((projected.x / projected.z, projected.y / projected.z))
        max_error = max(max_error, (uv_blender - uv_cv).length)
    if max_error > 0.02:
        raise ValueError(f"Blender/OpenCV calibration disagreement: {max_error} pixels")
    return max_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig", required=True)
    parser.add_argument("--output")
    parser.add_argument("--request")
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--hide-object", action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    rig = json.loads(Path(args.rig).read_text(encoding="utf-8"))
    scene = bpy.context.scene
    scene.frame_set(args.frame)
    for name in args.hide_object:
        if name not in bpy.data.objects:
            raise ValueError(f"unknown object to hide: {name}")
        bpy.data.objects[name].hide_render = True
    camera = bpy.data.objects.new(
        "CameraPlanning_External", bpy.data.cameras.new("CameraPlanning_External")
    )
    scene.collection.objects.link(camera)
    if not args.validate_only:
        if not args.output or not bpy.data.filepath:
            raise ValueError("rendering requires --output and a saved, frozen .blend snapshot")
        output = Path(args.output).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.film_transparent = False
        # Disable compositor and sequencer transforms that could invalidate K.
        scene.render.use_compositing = False
        scene.render.use_sequencer = False
    records = []
    if args.request:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if request.get("planning_mode") == "target_viewspace":
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from blender_target_visibility import MeshViewValidator
            validator = MeshViewValidator(request)
            checks = [{"camera_id": c["camera_id"], **validator.check(c)} for c in rig["cameras"]]
            if not args.validate_only:
                (output / "mesh_view_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
            if not all(c["valid"] for c in checks):
                raise ValueError("selected views failed actual-mesh validation; see mesh_view_validation.json")
    for spec in rig["cameras"]:
        error = configure(scene, camera, spec)
        record = {"camera_id": spec["camera_id"], "calibration_max_error_px": error}
        if not args.validate_only:
            name = str(spec["camera_id"])
            if Path(name).name != name or "/" in name or "\\" in name or ":" in name:
                raise ValueError("unsafe camera_id filename")
            path = output / f"{name}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            record.update(path=str(path), sha256=sha256(path))
        records.append(record)
    result = {
        "status": "calibration_only" if args.validate_only else "rendered",
        "blender_version": bpy.app.version_string,
        "rig_sha256": sha256(args.rig),
        "snapshot": bpy.data.filepath,
        "frame": args.frame,
        "hidden_objects": args.hide_object,
        "render_engine": scene.render.engine,
        "views": records,
    }
    if not args.validate_only:
        result["snapshot_sha256"] = sha256(bpy.data.filepath)
        (output / "render_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
