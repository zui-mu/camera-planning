import pytest
import yaml

from camera_planning.workflow import PilotConfig, load_pilot_config


def minimum_config(**updates):
    values = {
        "scene_id": "s",
        "source_scene": "scene.blend",
        "blender_executable": "blender.exe",
        "output": "run",
        "target_object": "Chair",
        "allowed_region_object": "CameraArea",
    }
    values.update(updates)
    return values


def test_pilot_requires_explicit_allowed_region():
    values = minimum_config()
    values["allowed_region_object"] = None
    with pytest.raises(ValueError, match="allowed_region"):
        PilotConfig.model_validate(values)


def test_rq1_requires_human_correspondence_acknowledgement():
    with pytest.raises(ValueError, match="reference-human"):
        PilotConfig.model_validate(minimum_config(reference_npz="ref.npz"))
    with pytest.raises(ValueError, match="confirm"):
        PilotConfig.model_validate(minimum_config(reference_npz="ref.npz", human_objects=["Human"]))


def test_imported_reference_mesh_is_declared_as_human():
    config = PilotConfig.model_validate(
        minimum_config(reference_mesh="human.ply", reference_object_name="FrozenHuman")
    )
    assert config.human_objects == ["FrozenHuman"]


def test_relative_paths_resolve_from_config(tmp_path):
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(minimum_config()), encoding="utf-8")
    config = load_pilot_config(path)
    assert config.source_scene == str((tmp_path / "scene.blend").resolve())
    assert config.output == str((tmp_path / "run").resolve())
