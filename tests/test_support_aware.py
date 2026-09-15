import json

import numpy as np

from camera_planning.contracts import Box, TargetViewSettings
from camera_planning.geometry import AABBGeometry
from camera_planning.support import build_interaction_region, detect_supported_objects
from camera_planning.workflow import PilotConfig, _write_box_request


def desk_scene():
    desk = Box(object_id="desk", minimum=(0.0, 0.0, 0.0), maximum=(2.0, 1.0, 0.8))
    lamp = Box(object_id="lamp", minimum=(0.2, 0.2, 0.8), maximum=(0.6, 0.6, 1.3))
    monitor = Box(
        object_id="monitor",
        minimum=(1.2, 0.35, 0.795),
        maximum=(1.7, 0.55, 1.25),
    )
    cabinet = Box(object_id="cabinet", minimum=(3.0, 0.0, 0.0), maximum=(4.0, 1.0, 2.0))
    return desk, lamp, monitor, cabinet


def test_support_detection_splits_three_geometric_roles():
    desk, lamp, monitor, cabinet = desk_scene()
    settings = TargetViewSettings()

    supported, remaining, graph = detect_supported_objects(
        desk, [lamp, monitor, cabinet], settings, up_index=2
    )

    assert [box.object_id for box in supported] == ["lamp", "monitor"]
    assert [box.object_id for box in remaining] == ["cabinet"]
    assert graph["supported_object_ids"] == ["lamp", "monitor"]
    assert all(item["confidence"] > 0 for item in graph["relations"])


def test_free_support_region_excludes_supported_object_footprints():
    desk, lamp, monitor, cabinet = desk_scene()
    settings = TargetViewSettings(
        interaction_anchor_count=100,
        interaction_height_offsets_m=[0.15, 0.5, 0.9],
        interaction_clearance_m=0.05,
    )
    region, manifest = build_interaction_region(
        desk, None, [lamp, monitor], [cabinet], settings, up_index=2
    )

    assert region is not None
    assert manifest["free_anchor_count"] > 0
    assert manifest["occupied_anchor_count"] > 0
    points = np.asarray(region.sample_points_world_m)
    for box in (lamp, monitor, cabinet):
        low = np.asarray(box.minimum) - settings.interaction_clearance_m
        high = np.asarray(box.maximum) + settings.interaction_clearance_m
        assert not np.any(np.all((points >= low) & (points <= high), axis=1))


def test_supported_objects_can_be_excluded_only_for_role_specific_rays():
    _, lamp, monitor, cabinet = desk_scene()
    geometry = AABBGeometry([lamp, monitor, cabinet])
    origin = np.array([-1.0, 0.4, 1.0])
    endpoints = np.array([[1.0, 0.4, 1.0], [3.5, 0.4, 1.0]])

    assert geometry.blocked(origin, endpoints).tolist() == [True, True]
    # Furniture-identity scoring may discount lamp/monitor occlusion, but the
    # cabinet remains a blocker.  Interaction rays call blocked() without this exclusion.
    assert geometry.blocked(
        origin, endpoints, exclude_object_ids={"lamp", "monitor"}
    ).tolist() == [False, True]


def test_box_workflow_writes_support_and_interaction_artifacts(tmp_path):
    desk, lamp, _, cabinet = desk_scene()
    manifest = {
        "target_asset": None,
        "boxes": [
            {"object_id": desk.object_id, "minimum_m": desk.minimum,
             "maximum_m": desk.maximum, "role": "target"},
            {"object_id": lamp.object_id, "minimum_m": lamp.minimum,
             "maximum_m": lamp.maximum, "role": "obstacle"},
            {"object_id": cabinet.object_id, "minimum_m": cabinet.minimum,
             "maximum_m": cabinet.maximum, "role": "obstacle"},
        ],
    }
    boxes_path = tmp_path / "scene_boxes.json"
    boxes_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = PilotConfig(
        scene_id="desk_unit",
        planning_mode="target_viewspace",
        source_scene="room.blend",
        target_scene="desk.usdc",
        blender_executable="blender.exe",
        output="unused",
        allowed_min_m=(-5.0, -5.0, 0.0),
        allowed_max_m=(5.0, 5.0, 3.0),
        path={"enabled": True},
    )
    request_path = tmp_path / "request.json"

    _write_box_request(config, boxes_path, request_path)

    request = json.loads(request_path.read_text(encoding="utf-8"))
    graph = json.loads((tmp_path / "support_graph.json").read_text(encoding="utf-8"))
    region = json.loads((tmp_path / "interaction_region.json").read_text(encoding="utf-8"))
    assert [item["object_id"] for item in request["supported_objects"]] == ["lamp"]
    assert [item["object_id"] for item in request["obstacles"]] == ["cabinet"]
    assert graph["supported_object_ids"] == ["lamp"]
    assert region["status"] == "ready"
    assert request["interaction_region"]["sample_points_world_m"]
