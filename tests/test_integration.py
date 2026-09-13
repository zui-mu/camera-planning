from pathlib import Path

import numpy as np
import pytest
import yaml

from camera_planning.evaluation import evaluate_fused
from camera_planning.experiments import prepare_rq1
from camera_planning.integration import command_plan
from camera_planning.planner import load_request, plan


def synthetic_reference(path, translation=0, topology="a" * 64):
    # Small metric fixture only; NOT a valid full MHR artifact for reconstruction.
    np.savez(
        path,
        format_version="mhr_fused_body_parameters_v1",
        profile="mhr_127_v1",
        length_unit="meter",
        mhr_asset_hash="b" * 64,
        checkpoint_hash="c" * 64,
        topology_hash=topology,
        joints_world=np.ones((127, 3)) * translation,
        root_rotation_world=np.eye(3),
    )


def test_fused_metrics_and_topology_guard(tmp_path):
    synthetic_reference(tmp_path / "ref.npz")
    synthetic_reference(tmp_path / "pred.npz", translation=1)
    result = evaluate_fused(tmp_path / "ref.npz", tmp_path / "pred.npz")
    assert result["root_aligned_mpjpe_mm"] == 0
    assert result["world_mpjpe_mm"] == pytest.approx(1000 * np.sqrt(3))
    synthetic_reference(tmp_path / "wrong.npz", topology="d" * 64)
    with pytest.raises(ValueError, match="topology_hash"):
        evaluate_fused(tmp_path / "ref.npz", tmp_path / "wrong.npz")


def test_experiment_and_command_paths(tmp_path):
    config = load_request(Path(__file__).parents[1] / "configs" / "demo_sofa.yaml")
    config.candidates.azimuth_count = 8
    config.observation.surface_samples = 24
    config.observation.envelope_samples = 24
    config.scoring.swap_passes = 0
    plan(config, tmp_path / "plan")
    synthetic_reference(tmp_path / "reference.npz")
    experiment = prepare_rq1(
        tmp_path / "plan", tmp_path / "experiment", tmp_path / "reference.npz", random_sets=2
    )
    assert len(experiment["sets"]) == 6
    pool = yaml.safe_load((tmp_path / "experiment" / "pool_cameras.json").read_text())
    used_ids = {camera_id for entry in experiment["sets"] for camera_id in entry["camera_ids"]}
    assert {camera["camera_id"] for camera in pool["cameras"]} == used_ids
    assert experiment["used_camera_count"] <= experiment["source_feasible_camera_count"]
    for entry in experiment["sets"]:
        scene = yaml.safe_load(
            (tmp_path / "experiment" / "sets" / entry["set_id"] / "scene.yaml").read_text()
        )
        assert scene["scene_id"] == config.scene_id
        assert len(scene["views"]) == config.budget
    commands = command_plan(
        tmp_path / "experiment",
        tmp_path / "ys",
        tmp_path / "python.exe",
        tmp_path / "body.yaml",
        tmp_path / "fuse.yaml",
    )
    assert commands["execution"] == "NOT_EXECUTED"
    fit = commands["commands"][0]["argv"]
    assert "sam3d_body" in fit
    assert Path(fit[fit.index("--output-dir") + 1]).name == "body_fit_pool"
    assert all("--project-root" in c["argv"] for c in commands["commands"])
    with pytest.raises(ValueError):
        command_plan(tmp_path / "experiment", tmp_path, "python", "body", "fuse")


def test_modified_plan_rejected_before_experiment(tmp_path):
    config = load_request(Path(__file__).parents[1] / "configs" / "demo_sofa.yaml")
    config.candidates.azimuth_count = 8
    config.observation.surface_samples = 24
    config.observation.envelope_samples = 24
    config.scoring.swap_passes = 0
    plan(config, tmp_path / "plan")
    synthetic_reference(tmp_path / "reference.npz")
    (tmp_path / "plan" / "evidence.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source plan changed"):
        prepare_rq1(tmp_path / "plan", tmp_path / "experiment", tmp_path / "reference.npz")
