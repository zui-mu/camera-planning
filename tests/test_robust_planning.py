import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from camera_planning.contracts import (
    Box,
    Intrinsics,
    ObservationSettings,
    ScoreSettings,
    TargetViewSettings,
)
from camera_planning.free_space import FreeGrid, furniture_domain
from camera_planning.geometry import AABBGeometry, look_at
from camera_planning.observability import lower_tail, point_quality, spatial_metrics
from camera_planning.planner import load_request, plan
from camera_planning.refinement import check_evidence_budget, concatenate_evidence
from camera_planning.scoring import build_evidence, evaluate_set
from camera_planning.selection import select
from camera_planning.target_viewspace import capture_box_for, views_are_distinct
from camera_planning.trajectory import connect, open_order


def spatial_evidence():
    cameras = [
        look_at(np.array(p, float), [0, 0, 0], [0, 1, 0], Intrinsics(), str(i))
        for i, p in enumerate(([0, 0, 3], [0, 0, -3], [3, 0, 0], [-3, 0, 0], [2, 2, 2], [-2, 2, 2]))
    ]
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [-0.2, -0.1, -0.1]])
    settings = ScoreSettings(objective="robust_space", multistarts=3)
    ev = build_evidence(
        cameras,
        points,
        np.zeros(3, int),
        AABBGeometry([]),
        settings,
        np.full(3, 2),
        weights=np.array([1.0, 2.0, 3.0]),
    )
    return ev, settings


def test_weighted_tail_partial_sample():
    assert lower_tail([0, 1], [0.1, 0.9], 0.2) == pytest.approx(0.5)
    assert lower_tail([0, 1], [0.1, 0.9], 1) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        lower_tail([0, 1], [0, 1], 0.2)


def test_directional_rank_and_monotonicity():
    ev, settings = spatial_evidence()
    assert point_quality(ev, [0], settings)[0] == 0
    assert point_quality(ev, [0, 1], settings)[0] == 0
    assert point_quality(ev, [0, 2], settings)[0] > 0
    assert np.all(
        point_quality(ev, range(6), settings) >= point_quality(ev, [0, 2], settings) - 1e-10
    )
    assert spatial_metrics(ev, [], settings)["robust_utility"] == 0


def test_lazy_matches_full_greedy_weighted_coverage():
    ev, settings = spatial_evidence()
    rng = np.random.default_rng(3)
    ev.visible = rng.random(ev.visible.shape) > 0.4
    a, _ = select(ev, 3, settings, "coverage")
    b, trace = select(ev, 3, settings, "lazy_coverage")
    assert a == b
    assert trace[-1]["objective"] == "weighted_single_view_coverage"


def test_robust_heuristic_vs_bounded_exact():
    ev, settings = spatial_evidence()
    exact, _ = select(ev, 3, settings, "exact")
    greedy, trace = select(ev, 3, settings, "greedy")
    assert (
        evaluate_set(ev, exact, settings)["total"]
        >= evaluate_set(ev, greedy, settings)["total"] - 1e-10
    )
    assert trace[-1]["score_evaluations"] <= settings.max_score_evaluations
    settings.exact_max_subsets = 1
    with pytest.raises(ValueError, match="capped"):
        select(ev, 3, settings, "exact")


def test_optional_milp_coverage_oracle():
    pytest.importorskip("scipy")
    ev, settings = spatial_evidence()
    ev.visible = np.random.default_rng(7).random(ev.visible.shape) > 0.5
    ids, trace = select(ev, 2, settings, "milp_coverage")
    weights = ev.weights / ev.weights.sum()
    actual = ev.visible[ids].any(axis=0) @ weights
    best = max(
        ev.visible[list(pair)].any(axis=0) @ weights for pair in itertools.combinations(range(6), 2)
    )
    assert actual == pytest.approx(best)
    assert trace[-1]["proven_optimal"]


def test_evidence_append_preserves_old_rows():
    ev, _ = spatial_evidence()
    before = ev.information[:3].copy()
    merged = concatenate_evidence(ev.subset([0, 1, 2]), ev.subset([3, 4, 5]))
    assert np.array_equal(before, merged.information[:3])
    assert np.array_equal(ev.visible, merged.visible)


def test_zero_prior_requires_observation_bounds():
    with pytest.raises(ValueError, match="observation.region"):
        ObservationSettings(mode="furniture_free_space")


def test_thin_wall_blocks_grid_edge_and_path():
    region = Box(object_id="room", minimum=(-1, -0.5, -0.5), maximum=(1, 0.5, 0.5))
    wall = Box(object_id="wall", minimum=(-0.01, -0.6, -0.6), maximum=(0.01, 0.6, 0.6))
    grid = FreeGrid(region, 0.5, AABBGeometry([wall]), 1000)
    assert grid.free.all()  # wall falls BETWEEN centers
    assert connect(grid, [-0.75, 0.25, 0.25], [0.75, 0.25, 0.25], 1000) is None


def test_capture_box_expansion_and_external_aabb_visibility():
    request = tiny_request()
    request.target_view.capture_side_padding_m = 0.5
    request.target_view.capture_top_padding_m = 2.0
    request.target_view.capture_bottom_padding_m = 0.25
    center, axes, half = capture_box_for(request)
    corners = center + np.array(list(itertools.product((-1, 1), repeat=3))) * half @ axes.T
    target_low, target_high = np.asarray(request.target.minimum), np.asarray(request.target.maximum)
    assert corners[:, request.up_index].min() == pytest.approx(
        target_low[request.up_index] - 0.25
    )
    assert corners[:, request.up_index].max() == pytest.approx(
        target_high[request.up_index] + 2.0
    )

    target_only = AABBGeometry([request.target])
    origin = np.asarray(request.target.minimum) - 1
    endpoint = np.asarray(request.target.maximum) + 1
    assert target_only.blocked(origin, [endpoint])[0]
    assert not target_only.blocked(
        origin, [endpoint], exclude_object_id=request.target.object_id
    )[0]


def test_view_pair_requires_angle_and_position_floors():
    settings = TargetViewSettings(
        min_view_direction_separation_degrees=6,
        min_camera_position_separation_m=0.35,
    )

    def record(position):
        position = np.asarray(position, float)
        return {
            "camera": look_at(position, [0, 0, 0], [0, 0, 1], Intrinsics(), "test"),
            "direction": position / np.linalg.norm(position),
        }

    assert not views_are_distinct(record([3, 0, 1]), record([3.05, 0, 1]), settings)
    assert not views_are_distinct(record([3, 0, 1]), record([4, 0, 4 / 3]), settings)
    assert views_are_distinct(record([3, 0, 1]), record([3, 0.8, 1.2]), settings)


def test_path_detours_and_open_order():
    region = Box(object_id="room", minimum=(-2, -0.5, -2), maximum=(2, 0.5, 2))
    obstacle = Box(object_id="box", minimum=(-0.3, -0.6, -0.3), maximum=(0.3, 0.6, 0.3))
    grid = FreeGrid(region, 0.25, AABBGeometry([obstacle]), 5000, 0.05)
    path = connect(grid, [-1, 0, 0], [1, 0, 0], 5000)
    assert path and len(path) > 2
    assert all(grid.segment_clear(a, b) for a, b in itertools.pairwise(path))
    costs = np.array([[0, 1, 10], [1, 0, 1], [10, 1, 0]])
    order = open_order(costs)
    assert sum(costs[a, b] for a, b in itertools.pairwise(order)) == 2
    assert not grid.segment_clear([-3, 0, 0], [-1, 0, 0])


def tiny_request():
    request = load_request(Path(__file__).parents[1] / "configs/robust_sofa_demo.yaml")
    request.observation.voxel_size_m = 0.4
    request.observation.spatial_samples = 48
    request.observation.surface_samples = 24
    request.candidates.azimuth_count = 8
    request.budget = 3
    request.adaptive.rounds = 1
    request.adaptive.proposals_per_round = 8
    request.scoring.multistarts = 1
    request.scoring.swap_passes = 1
    request.path.enabled = False
    return request


def test_domain_ignores_legacy_human_height_and_is_in_room():
    request = tiny_request()
    geometry = AABBGeometry([request.target, *request.obstacles])
    a, weights, meta = furniture_domain(request, geometry)
    request.observation.full_body_height_m = 9
    b, _, _ = furniture_domain(request, geometry)
    assert np.array_equal(a, b)
    assert not geometry.occupied(a).any()
    assert np.all(a >= request.observation.region.minimum)
    assert np.all(a <= request.observation.region.maximum)
    assert weights.sum() == pytest.approx(meta["estimated_volume_m3"])
    assert not meta["human_scale_used"]
    request.adaptive.max_evidence_mb = 0.01
    with pytest.raises(ValueError, match="evidence arrays"):
        check_evidence_budget(request, 100, 1000)


def test_adaptive_plan_artifacts_and_incumbent(tmp_path):
    request = tiny_request()
    result = plan(request, tmp_path / "modern")
    assert result["budget"] == 3
    assert result["observation_domain"] == "scene_constrained_furniture_free_space_v1"
    assert result["candidate_pool_reference"]["hard_regions_removed_from_objective"] is False
    for entry in result["adaptive_trace"]:
        if "score_after" in entry:
            assert entry["score_after"] >= entry["score_before"]
            assert entry["old_evidence_rows_recomputed"] == 0
    with np.load(tmp_path / "modern/evidence.npz") as archive:
        assert "weights" in archive.files and 2 in archive["kinds"]
    rig = json.loads((tmp_path / "modern/cameras.json").read_text())
    assert set(rig) == {"scene_id", "cameras"}
    from camera_planning.cached_selection import load_cached_evidence, select_cached

    cached = select_cached(tmp_path / "modern", tmp_path / "baseline", "lazy_coverage")
    assert cached["geometry_queries"] == cached["render_calls"] == 0
    rig["cameras"][0]["camera_id"] = "tampered"
    pool_file = tmp_path / "modern/candidate_cameras.json"
    pool_file.write_text(json.dumps(rig))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cached_evidence(tmp_path / "modern")


def test_cli_imports():
    from camera_planning.cli import main

    assert callable(main)


def test_adaptive_can_bootstrap_too_small_pool(monkeypatch):
    import camera_planning.refinement as module

    ev, _ = spatial_evidence()
    request = tiny_request()
    monkeypatch.setattr(module, "propose", lambda *args: iter([(ev.cameras[2], "test")]))
    monkeypatch.setattr(module, "pose_rejection", lambda *args: None)
    monkeypatch.setattr(module, "filter_evidence", lambda request, evidence, rejected: evidence)
    merged, indices, trace, history = module.refine(
        request,
        AABBGeometry([]),
        ev.subset([0, 1]),
        [],
        "greedy",
        0,
    )
    assert len(indices) == len(merged.cameras) == 3
    assert trace[0]["event"] == "bootstrap_incomplete_pool"
    assert not history[0]["complete_set_before"] and history[0]["complete_set_after"]
