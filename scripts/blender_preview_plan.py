"""Create a non-destructive Blender copy containing candidate/selected cameras."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def camera_world(spec):
    rotation = Matrix(spec["R"])
    center = -(rotation.transposed() @ Vector(spec["T"]))
    matrix = (rotation.transposed() @ Matrix.Diagonal((1.0, -1.0, -1.0))).to_4x4()
    matrix.translation = center
    return matrix


def add_box(collection, minimum, maximum, name="CameraPlan_TargetBounds", color=(1.0, 0.55, 0.05, 1.0)):
    minimum, maximum = np.array(minimum), np.array(maximum)
    vertices = [
        (x, y, z)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges, [])
    obj = bpy.data.objects.new(name, mesh)
    obj.color = color
    collection.objects.link(obj)


def add_points(collection, name, points, color):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points.tolist(), [], [])
    obj = bpy.data.objects.new(name, mesh)
    obj.color = color
    # Respect depth: drawing through the coffee table made exterior samples look embedded.
    obj.show_in_front = False
    obj.hide_render = True
    collection.objects.link(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-rig", required=True)
    parser.add_argument("--candidate-rig")
    parser.add_argument("--request")
    parser.add_argument("--evidence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not bpy.data.filepath:
        raise ValueError("preview requires a saved source .blend")
    if "CameraPlanning_Preview" in bpy.data.collections:
        raise ValueError("source scene already contains CameraPlanning_Preview")
    selected = json.loads(Path(args.selected_rig).read_text(encoding="utf-8"))
    candidates = (
        json.loads(Path(args.candidate_rig).read_text(encoding="utf-8"))
        if args.candidate_rig
        else selected
    )
    selected_ids = {camera["camera_id"] for camera in selected["cameras"]}
    root = bpy.data.collections.new("CameraPlanning_Preview")
    bpy.context.scene.collection.children.link(root)
    selected_collection = bpy.data.collections.new("Selected_Cameras")
    candidate_collection = bpy.data.collections.new("Unselected_Candidates")
    diagnostics = bpy.data.collections.new("Planning_Diagnostics")
    root.children.link(selected_collection)
    root.children.link(candidate_collection)
    root.children.link(diagnostics)
    active = None
    for spec in candidates["cameras"]:
        data = bpy.data.cameras.new(f"Plan_{spec['camera_id']}")
        data.display_size = 0.25
        data.show_name = True
        k, width = spec["K"], spec["width"]
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36.0
        data.lens = k[0][0] * data.sensor_width / width
        obj = bpy.data.objects.new(f"Plan_{spec['camera_id']}", data)
        obj.matrix_world = camera_world(spec)
        obj.show_name = True
        obj["camera_id"] = spec["camera_id"]
        obj["selected"] = spec["camera_id"] in selected_ids
        obj.color = (0.05, 0.8, 0.25, 1.0) if obj["selected"] else (0.35, 0.38, 0.42, 1.0)
        (selected_collection if obj["selected"] else candidate_collection).objects.link(obj)
        if active is None and obj["selected"]:
            active = obj
    if args.request:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        add_box(diagnostics, request["target"]["minimum"], request["target"]["maximum"])
        for index, box in enumerate(request.get("supported_objects", [])):
            add_box(
                diagnostics,
                box["minimum"],
                box["maximum"],
                f"CameraPlan_SupportedObject_{index:02d}",
                (0.58, 0.45, 0.95, 1.0),
            )
    if args.evidence:
        with np.load(args.evidence, allow_pickle=False) as evidence:
            points = evidence["points"].copy()
            kinds = evidence["kinds"].copy() if "kinds" in evidence.files else np.ones(len(points))
        if np.any(kinds == 0):
            add_points(
                diagnostics,
                "CameraPlan_FurnitureSurfaceSamples",
                points[kinds == 0],
                (1.0, 0.45, 0.05, 1.0),
            )
        if np.any(kinds == 3):
            add_points(diagnostics, "CameraPlan_TargetFramingCorners", points[kinds == 3],
                       (1.0, 0.55, 0.05, 1.0))
        if np.any(kinds == 1):
            add_points(
                diagnostics,
                "CameraPlan_FramingEnvelopeSamples",
                points[kinds == 1],
                (0.1, 0.55, 1.0, 1.0),
            )
        if np.any(kinds == 2):
            quality_path = Path(args.evidence).parent / "observability.npz"
            if quality_path.is_file():
                with np.load(quality_path, allow_pickle=False) as data:
                    quality = data["selected_quality"]
                # Absolute score bins, not a percentile trick that always paints some green.
                for label, mask, color in (
                    ("Weak", quality < 0.2, (1.0, 0.15, 0.1, 1.0)),
                    ("Medium", (quality >= 0.2) & (quality < 0.6), (1.0, 0.7, 0.1, 1.0)),
                    ("Strong", quality >= 0.6, (0.1, 0.8, 0.3, 1.0)),
                ):
                    if mask.any():
                        add_points(
                            diagnostics,
                            f"CameraPlan_Space_{label}",
                            points[kinds == 2][mask],
                            color,
                        )
            else:
                add_points(
                    diagnostics, "CameraPlan_FreeSpace", points[kinds == 2], (0.1, 0.55, 1.0, 1.0)
                )
        if np.any(kinds == 4):
            add_points(
                diagnostics,
                "CameraPlan_FreeSupportAnchors",
                points[kinds == 4],
                (0.05, 0.75, 0.42, 1.0),
            )
        if np.any(kinds == 5):
            add_points(
                diagnostics,
                "CameraPlan_OccupiedSupportAnchors",
                points[kinds == 5],
                (0.95, 0.15, 0.12, 1.0),
            )
        if np.any(kinds == 6):
            add_points(
                diagnostics,
                "CameraPlan_UnknownHumanInteractionSamples",
                points[kinds == 6],
                (0.10, 0.50, 1.0, 1.0),
            )
        path_file = Path(args.evidence).parent / "camera_path.json"
        if path_file.is_file():
            path_data = json.loads(path_file.read_text(encoding="utf-8"))
            if path_data.get("status") == "planned":
                vertices, edges = [], []
                for leg in path_data["legs"]:
                    offset = len(vertices)
                    vertices.extend(leg["points"])
                    edges.extend(
                        (offset + i, offset + i + 1) for i in range(len(leg["points"]) - 1)
                    )
                mesh = bpy.data.meshes.new("CameraPlan_OpenPath")
                mesh.from_pydata(vertices, edges, [])
                obj = bpy.data.objects.new("CameraPlan_OpenPath", mesh)
                obj.hide_render = True
                diagnostics.objects.link(obj)
    if active is not None:
        bpy.context.scene.camera = active
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(
        json.dumps(
            {
                "status": "saved",
                "preview": str(output),
                "selected_count": len(selected_ids),
                "candidate_count": len(candidates["cameras"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
