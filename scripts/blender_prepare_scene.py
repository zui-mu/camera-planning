"""Import a supported 3D scene and save a frozen .blend copy.

Run with Blender --background --factory-startup --python this_file -- ...
Houdini .hip is intentionally unsupported: it must be exported by Houdini first.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalize_imported_image_paths(source_parent):
    """Make USD-relative texture references stable in the frozen .blend.

    Blender's USD importer resolves a relative asset against the process
    working directory.  The pilot therefore starts Blender in the USD's
    directory, then stores the resolved absolute path in the snapshot so
    later render stages can run from any directory.
    """
    parent = Path(source_parent).resolve()
    for image in bpy.data.images:
        if image.source != "FILE" or image.packed_file or not image.filepath:
            continue
        raw = image.filepath.replace("\\", "/")
        candidates = []
        if raw.startswith("//"):
            candidates.append(parent / raw[2:])
        else:
            candidate = Path(raw)
            if candidate.is_absolute():
                candidates.append(candidate)
            else:
                candidates.append(parent / raw)
                candidates.append(parent / raw.lstrip("./"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                image.filepath = str(resolved)
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-mesh")
    parser.add_argument("--reference-object-name", default="ReferenceHuman")
    parser.add_argument("--exclude-object", action="append", default=[])
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
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
    elif suffix in {".hip", ".hiplc", ".hipnc"}:
        raise ValueError("Houdini scenes require a licensed Houdini export to USD/FBX/GLB first")
    else:
        raise ValueError(f"unsupported scene format: {suffix}")
    normalize_imported_image_paths(source.parent)
    for name in args.exclude_object:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = True
            obj.hide_set(True)
    reference_record = None
    if args.reference_mesh:
        reference = Path(args.reference_mesh).resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
        if args.reference_object_name in bpy.data.objects:
            raise ValueError(f"reference object name already exists: {args.reference_object_name}")
        before = set(bpy.data.objects)
        reference_suffix = reference.suffix.lower()
        if reference_suffix == ".ply":
            bpy.ops.wm.ply_import(filepath=str(reference))
        elif reference_suffix == ".obj":
            bpy.ops.wm.obj_import(filepath=str(reference))
        else:
            raise ValueError("reference mesh import supports .ply or .obj")
        created = [obj for obj in bpy.data.objects if obj not in before]
        meshes = [obj for obj in created if obj.type == "MESH"]
        if len(meshes) != 1:
            raise ValueError("reference mesh must import as exactly one mesh object")
        human = meshes[0]
        human.name = args.reference_object_name
        # Fused MHR PLY is already in world meters. Do not guess or apply an
        # axis/scale/translation correction merely to make it look plausible.
        if not human.data.materials:
            material = bpy.data.materials.new("CameraPlanning_ReferenceNeutral")
            material.diffuse_color = (0.55, 0.42, 0.36, 1.0)
            human.data.materials.append(material)
        reference_record = {
            "path": str(reference),
            "sha256": digest(reference),
            "object_name": human.name,
            "matrix_world": [list(row) for row in human.matrix_world],
            "transform_policy": "identity_import_world_coordinates_no_auto_alignment",
        }
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    manifest = {
        "format_version": "camera_blender_snapshot_v1",
        "source": str(source),
        "source_sha256": digest(source),
        "snapshot": str(output),
        "snapshot_sha256": digest(output),
        "blender_version": bpy.app.version_string,
        "mesh_objects": sorted(obj.name for obj in bpy.data.objects if obj.type == "MESH"),
        "reference_mesh": reference_record,
        "excluded_objects": args.exclude_object,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
