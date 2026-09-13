import json

from camera_planning.workflow import PilotConfig, _box_export_arguments, _csv, _write_box_request


def test_negative_vector_serialization_can_be_used_with_equals_syntax():
    assert "--allowed-min=" + _csv((-0.75, -0.5, 0.2)) == "--allowed-min=-0.75,-0.5,0.2"


def test_box_manifest_becomes_aabb_planning_request(tmp_path):
    config = PilotConfig.model_validate(
        {
            "scene_id": "s",
            "source_scene": "scene.blend",
            "blender_executable": "blender.exe",
            "output": "run",
            "target_object": "Chair",
            "allowed_region_object": "CameraArea",
            "floor_height_m": 0,
        }
    )
    boxes = {
        "boxes": [
            {
                "object_id": "Chair",
                "minimum_m": [-1, 0, 0],
                "maximum_m": [1, 1, 1],
                "role": "target",
            },
            {
                "object_id": "Table",
                "minimum_m": [2, 0, 0],
                "maximum_m": [3, 1, 1],
                "role": "obstacle",
            },
            {
                "object_id": "CameraArea",
                "minimum_m": [-5, -5, 0.1],
                "maximum_m": [5, 5, 3],
                "role": "allowed_region",
            },
        ]
    }
    source = tmp_path / "boxes.json"
    source.write_text(json.dumps(boxes), encoding="utf-8")
    output = tmp_path / "request.json"
    _write_box_request(config, source, output)
    request = json.loads(output.read_text(encoding="utf-8"))
    assert request["geometry"] == {"backend": "aabb", "mesh_file": None, "sha256": None}
    assert request["target"]["object_id"] == "Chair"
    assert [item["object_id"] for item in request["obstacles"]] == ["Table"]


def test_dual_scene_input_does_not_require_a_target_name():
    config = PilotConfig.model_validate(
        {
            "scene_id": "s",
            "source_scene": "room.usdc",
            "target_scene": "sofa.usdc",
            "blender_executable": "blender.exe",
            "output": "run",
            "allowed_min_m": [0, 0, 0.2],
            "allowed_max_m": [5, 5, 2.5],
        }
    )
    arguments = _box_export_arguments(config, "boxes.json", "target_bounds.json")
    assert "--target-object" not in arguments
    assert arguments[arguments.index("--target-bounds") + 1] == "target_bounds.json"
    assert arguments[arguments.index("--structure-policy") + 1] == "auto_geometry"
