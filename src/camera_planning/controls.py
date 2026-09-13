"""Explicit RQ1 control, never a measured scene-consistency score or video fallback."""

import json
from pathlib import Path

from .artifacts import file_hash, new_output, write_json


def prepare_constant_scene_control(experiment_dir, human_objects, acknowledge_assumption=False):
    if not acknowledge_assumption:
        raise ValueError(
            "requires explicit acknowledgement: r_scene=1 is an experimental assumption"
        )
    if not human_objects:
        raise ValueError("name the reference-human mesh objects hidden in the empty render")
    root = Path(experiment_dir).resolve()
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if file_hash(experiment["reference"]["path"]) != experiment["reference"]["sha256"]:
        raise ValueError("reference artifact changed since experiment preparation")
    human = json.loads(
        (root / "renders" / "human" / "render_manifest.json").read_text(encoding="utf-8")
    )
    empty = json.loads(
        (root / "renders" / "empty" / "render_manifest.json").read_text(encoding="utf-8")
    )
    if human["status"] != "rendered" or empty["status"] != "rendered":
        raise ValueError("both image sets must actually be rendered")
    for key in ("snapshot_sha256", "rig_sha256", "frame", "render_engine", "blender_version"):
        if human[key] != empty[key]:
            raise ValueError(f"controlled render mismatch: {key}")
    if human["rig_sha256"] != file_hash(root / "pool_cameras.json"):
        raise ValueError("rendered rig is not this experiment's candidate pool")
    if set(empty["hidden_objects"]) != set(human["hidden_objects"]) | set(human_objects):
        raise ValueError(
            "empty/human render visibility difference must equal the declared human objects"
        )
    if set(human["hidden_objects"]) & set(human_objects):
        raise ValueError("reference human was hidden in the human render")
    expected = {
        c["camera_id"] for c in json.loads((root / "pool_cameras.json").read_text())["cameras"]
    }
    records = []
    for label, manifest in (("human", human), ("empty", empty)):
        if {v["camera_id"] for v in manifest["views"]} != expected:
            raise ValueError(f"{label} views do not cover the candidate pool")
        for record in manifest["views"]:
            path = root / "renders" / label / f"{record['camera_id']}.png"
            if file_hash(path) != record["sha256"]:
                raise ValueError(f"render image changed: {path}")
            if record["calibration_max_error_px"] > 0.02:
                raise ValueError("render calibration did not pass")
    destination = new_output(root / "scene_consistency_pool")
    for camera_id in sorted(expected):
        path = destination / camera_id / "scene_consistency_result.json"
        record = {
            "module": "scene_consistency",
            "scene_id": experiment["scene_id"],
            "view_id": camera_id,
            "empty_image": str(root / "renders" / "empty" / f"{camera_id}.png"),
            "generated_image": str(root / "renders" / "human" / f"{camera_id}.png"),
            "backend": "precomputed",
            "reference_mode": "precomputed",
            "r_scene": 1.0,
            "object_metrics": [],
            "exclude_body_mask": False,
            "body_mask_mode": "disabled",
            "mask_source": None,
            "dilate_radius_px": 0,
            "status": "ok",
            "outputs": {"result": str(path)},
            "warnings": [
                "CONTROLLED ASSUMPTION: r_scene=1 was not measured; do not use for generated-video evaluation."
            ],
            "metadata": {
                "protocol": "rq1_assumed_constant_scene_quality_v1",
                "is_measured_scene_score": False,
                "reference_sha256": experiment["reference"]["sha256"],
                "snapshot_sha256": human["snapshot_sha256"],
                "reference_mesh_correspondence": "requires_user_verification",
            },
        }
        write_json(path, record)
        records.append(str(path))
    # Preserve experiment.json; the explicit protocol choice is a separate audit artifact.
    result = {
        "protocol": "rq1_assumed_constant_scene_quality_v1",
        "is_measured_scene_score": False,
        "records": records,
        "human_objects": list(human_objects),
    }
    write_json(root / "scene_control_manifest.json", result)
    return result
