import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from camera_planning.contracts import Box, Intrinsics, PlanningRequest, ScoreSettings
from camera_planning.geometry import (
    AABBGeometry,
    blender_camera_matrix,
    look_at,
    project,
    projection_jacobian,
)
from camera_planning.planner import load_request, plan, prepare
from camera_planning.scoring import build_evidence, evaluate_set
from camera_planning.selection import select
from camera_planning.evaluation import joint_metrics
from camera_planning.artifacts import new_output


@pytest.fixture
def request_config():
    config = load_request(Path(__file__).parents[1] / "configs" / "demo_sofa.yaml")
    config.candidates.azimuth_count = 8
    config.observation.surface_samples = 48
    config.observation.envelope_samples = 48
    config.scoring.swap_passes = 1
    return config


@pytest.mark.parametrize("up", [[0, 1, 0], [0, 0, 1]])
def test_look_at_and_blender(up):
    c = look_at(np.array([3.0, 2.0, 4.0]), np.array([0.0, 1.0, 0.0]), up, Intrinsics(), "a")
    uv, depth = project(c, [[0, 1, 0]])
    assert depth[0] > 0
    assert np.allclose(uv, [[512, 512]])
    assert np.allclose(c.center, [3, 2, 4])
    matrix = blender_camera_matrix(c)
    assert np.allclose(matrix[:3, :3] @ [0, 0, -1], np.array(c.R)[2])
    assert np.allclose(matrix[:3, :3] @ [0, 1, 0], -np.array(c.R)[1])


def test_jacobian_finite_difference():
    c = look_at(np.array([3.0, 2.0, 4.0]), [0, 1, 0], [0, 1, 0], Intrinsics(), "a")
    points = np.array([[0.2, 0.8, 0.3], [-0.1, 1.2, -0.2]])
    analytic = projection_jacobian(c, points)
    for axis in range(3):
        delta = np.eye(3)[axis] * 1e-5
        numeric = (project(c, points + delta)[0] - project(c, points - delta)[0]) / 2e-5
        assert np.allclose(analytic[:, :, axis], numeric, atol=1e-5)


@pytest.mark.parametrize(
    "origin,end,blocked",
    [
        ([-2, 0, 0], [2, 0, 0], True),
        ([-2, 2, 0], [2, 2, 0], False),
        ([2, 0, 0], [3, 0, 0], False),
        ([0, 0, 0], [2, 0, 0], True),
        ([-2, 0, 0], [-1, 0, 0], False),
        ([0, 2, 0], [0, -2, 0], True),
    ],
)
def test_segment_boxes(origin, end, blocked):
    box = Box(object_id="b", minimum=(-1, -1, -1), maximum=(1, 1, 1))
    backend = AABBGeometry([box])
    assert bool(backend.blocked(np.array(origin), np.array([end]))[0]) is blocked


def test_empty_geometry():
    backend = AABBGeometry([])
    assert not backend.occupied(np.zeros((2, 3))).any()
    assert not backend.blocked(np.ones(3), np.zeros((2, 3))).any()


def test_aabb_surface_sampling_excludes_underside():
    backend = AABBGeometry([Box(object_id="b", minimum=(-1, 0, -1), maximum=(1, 1, 1))])
    points, normals = backend.sample_surface("b", 200, 4, 1, -0.15)
    assert points.shape == normals.shape == (200, 3)
    assert np.all(normals[:, 1] >= -0.15)


def test_opposed_collinear_not_good_parallax():
    settings = ScoreSettings()
    cameras = [
        look_at(np.array(p, float), [0, 0, 0], [0, 1, 0], Intrinsics(), str(i))
        for i, p in enumerate(([0, 0, 3], [0, 0, -3], [3, 0, 0]))
    ]
    ev = build_evidence(cameras, np.zeros((1, 3)), np.zeros(1, int), AABBGeometry([]), settings)
    assert evaluate_set(ev, [0, 1], settings)["multiview"] == 0
    assert evaluate_set(ev, [0, 2], settings)["multiview"] == 1


@pytest.mark.parametrize("method", ["greedy", "random", "uniform", "farthest", "coverage"])
def test_fixed_budget_reproducibility(request_config, method):
    ev, _ = prepare(request_config)
    a, _ = select(ev, 4, request_config.scoring, method, 7, 1)
    b, _ = select(ev, 4, request_config.scoring, method, 7, 1)
    assert len(set(a)) == 4 and a == b
    with pytest.raises(ValueError):
        select(ev, len(ev.cameras) + 1, request_config.scoring)


def test_exact_at_least_greedy(request_config):
    ev, _ = prepare(request_config)
    ev = ev.subset(list(range(6)))
    a, _ = select(ev, 3, request_config.scoring, "exact")
    b, _ = select(ev, 3, request_config.scoring, "greedy")
    assert (
        evaluate_set(ev, a, request_config.scoring)["total"]
        >= evaluate_set(ev, b, request_config.scoring)["total"] - 1e-10
    )


def test_request_cannot_accept_reference(request_config):
    data = request_config.model_dump()
    data["reference_human"] = "forbidden.npz"
    with pytest.raises(ValidationError):
        PlanningRequest.model_validate(data)
    data.pop("reference_human")
    data["length_unit"] = "cm"
    with pytest.raises(ValidationError):
        PlanningRequest.model_validate(data)


def test_joint_translation_is_not_pose_error():
    reference = np.arange(30).reshape(10, 3) / 100
    result = joint_metrics(reference, reference + [1, 0, 0], 1)
    assert result["world_mpjpe_mm"] == pytest.approx(1000)
    assert result["root_aligned_mpjpe_mm"] == pytest.approx(0, abs=1e-10)


def test_plan_artifacts(request_config, tmp_path):
    result = plan(request_config, tmp_path / "plan")
    exported = json.loads((tmp_path / "plan" / "cameras.json").read_text())
    placements = json.loads((tmp_path / "plan" / "camera_placements.json").read_text())
    assert len(exported["cameras"]) == request_config.budget
    assert len(placements["cameras"]) == request_config.budget
    assert len(placements["cameras"][0]["position_world_m"]) == 3
    assert set(exported) == {"scene_id", "cameras"}
    assert result["reconstruction_status"] == "not_run"
    assert result["observation_domain"].startswith("pose_agnostic_furniture")
    with pytest.raises(FileExistsError):
        new_output(tmp_path / "plan")


def test_duplicate_candidate_layers_do_not_duplicate_poses(request_config):
    request_config.candidates.shell_offsets_m = [2.0, 2.0]
    evidence, rejected = prepare(request_config)
    centers = [tuple(np.round(c.center, 7)) for c in evidence.cameras]
    assert len(centers) == len(set(centers))
    assert any(r["reason"] == "duplicate_pose" for r in rejected)
