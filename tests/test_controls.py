import json

import pytest

from camera_planning.artifacts import file_hash, write_json
from camera_planning.controls import prepare_constant_scene_control


def fixture_renders(root):
    # Content hashes exercise provenance gates only; these are not real render images.
    reference = root / "reference.txt"
    reference.write_text("test reference")
    write_json(
        root / "experiment.json",
        {"scene_id": "test", "reference": {"path": str(reference), "sha256": file_hash(reference)}},
    )
    write_json(root / "pool_cameras.json", {"cameras": [{"camera_id": "cam_0"}]})
    for label in ("human", "empty"):
        directory = root / "renders" / label
        directory.mkdir(parents=True)
        path = directory / "cam_0.png"
        path.write_bytes(b"test image placeholder")
        write_json(
            directory / "render_manifest.json",
            {
                "status": "rendered",
                "snapshot_sha256": "s",
                "frame": 1,
                "render_engine": "test",
                "blender_version": "test",
                "rig_sha256": file_hash(root / "pool_cameras.json"),
                "hidden_objects": [] if label == "human" else ["RefHuman"],
                "views": [
                    {"camera_id": "cam_0", "sha256": file_hash(path), "calibration_max_error_px": 0}
                ],
            },
        )


def test_control_requires_acknowledgement(tmp_path):
    with pytest.raises(ValueError, match="acknowledgement"):
        prepare_constant_scene_control(tmp_path, ["RefHuman"])


def test_explicit_control_records_assumption(tmp_path):
    fixture_renders(tmp_path)
    result = prepare_constant_scene_control(tmp_path, ["RefHuman"], True)
    record = json.loads(
        (
            tmp_path / "scene_consistency_pool" / "cam_0" / "scene_consistency_result.json"
        ).read_text()
    )
    assert not result["is_measured_scene_score"]
    assert not record["metadata"]["is_measured_scene_score"]
    assert record["backend"] == "precomputed"


def test_control_rejects_mutated_images(tmp_path):
    fixture_renders(tmp_path)
    (tmp_path / "renders" / "human" / "cam_0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        prepare_constant_scene_control(tmp_path, ["RefHuman"], True)
